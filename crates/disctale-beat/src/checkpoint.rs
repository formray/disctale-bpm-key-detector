//! Reads the official Beat This! checkpoint as PyTorch writes it: a ZIP of
//! stored (uncompressed) entries holding `data.pkl`, a pickle of the
//! Lightning checkpoint, and one raw little-endian file per tensor storage.
//!
//! Only what that file uses is understood: the opcodes of pickle protocol 2
//! that `torch.save` emits for a state dict, and the four globals it names
//! (`OrderedDict`, `_rebuild_tensor_v2`, `FloatStorage`, `LongStorage`).
//! Anything else is an error, never a guess: a checkpoint we cannot read
//! fully must not run.

use std::collections::HashMap;

#[derive(Debug, thiserror::Error)]
pub enum CheckpointError {
    #[error("not a readable checkpoint archive: {0}")]
    Archive(String),
    #[error("unsupported pickle content: {0}")]
    Pickle(String),
    #[error("tensor {0} is missing")]
    Missing(String),
    #[error("tensor {name} has shape {found:?}, expected {expected:?}")]
    Shape {
        name: String,
        found: Vec<usize>,
        expected: Vec<usize>,
    },
}

/// The float tensors of the model, by their name without the `model.` prefix.
#[derive(Debug, Default)]
pub struct Weights {
    tensors: HashMap<String, (Vec<usize>, Vec<f32>)>,
}

impl Weights {
    pub fn take(&mut self, name: &str, shape: &[usize]) -> Result<Vec<f32>, CheckpointError> {
        let (found, data) = self
            .tensors
            .remove(name)
            .ok_or_else(|| CheckpointError::Missing(name.to_string()))?;
        if found != shape {
            return Err(CheckpointError::Shape {
                name: name.to_string(),
                found,
                expected: shape.to_vec(),
            });
        }
        Ok(data)
    }

    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.tensors.keys()
    }
}

pub fn read(bytes: &[u8]) -> Result<Weights, CheckpointError> {
    let entries = zip_entries(bytes)?;
    let (pkl_name, pkl) = entries
        .iter()
        .find(|(name, _)| name.ends_with("/data.pkl"))
        .ok_or_else(|| CheckpointError::Archive("no data.pkl".into()))?;
    let prefix = pkl_name.trim_end_matches("data.pkl");
    if let Some((_, order)) = entries
        .iter()
        .find(|(name, _)| name.ends_with("/byteorder"))
        && *order != b"little"
    {
        return Err(CheckpointError::Archive(
            "storages are not little-endian".into(),
        ));
    }
    let root = unpickle(pkl)?;
    let state = dict_get(&root, "state_dict")
        .ok_or_else(|| CheckpointError::Pickle("no state_dict".into()))?;
    let Value::Dict(items) = state else {
        return Err(CheckpointError::Pickle("state_dict is not a dict".into()));
    };
    let mut weights = Weights::default();
    for (key, value) in items {
        let Value::Str(name) = key else {
            return Err(CheckpointError::Pickle("non-string state_dict key".into()));
        };
        let Value::Tensor {
            storage,
            dtype,
            offset,
            size,
            stride,
        } = value
        else {
            return Err(CheckpointError::Pickle(format!("{name} is not a tensor")));
        };
        // Batch-norm step counters are int64 and unused at inference.
        if dtype != "FloatStorage" {
            continue;
        }
        let file = format!("{prefix}data/{storage}");
        let raw = entries
            .iter()
            .find(|(entry, _)| *entry == file)
            .map(|(_, data)| *data)
            .ok_or_else(|| CheckpointError::Archive(format!("no storage {file}")))?;
        let data = gather(raw, *offset, size, stride)
            .ok_or_else(|| CheckpointError::Archive(format!("{name} overruns its storage")))?;
        let short = name.strip_prefix("model.").unwrap_or(name).to_string();
        weights.tensors.insert(short, (size.clone(), data));
    }
    Ok(weights)
}

fn gather(raw: &[u8], offset: usize, size: &[usize], stride: &[usize]) -> Option<Vec<f32>> {
    let count: usize = size.iter().product();
    let mut out = Vec::with_capacity(count);
    let mut index = vec![0usize; size.len()];
    for _ in 0..count {
        let element = offset + index.iter().zip(stride).map(|(i, s)| i * s).sum::<usize>();
        let at = element.checked_mul(4)?;
        let word = raw.get(at..at + 4)?;
        out.push(f32::from_le_bytes([word[0], word[1], word[2], word[3]]));
        for axis in (0..size.len()).rev() {
            index[axis] += 1;
            if index[axis] < size[axis] {
                break;
            }
            index[axis] = 0;
        }
    }
    Some(out)
}

// ---------------------------------------------------------------- zip --

fn u16_at(b: &[u8], at: usize) -> Option<u64> {
    Some(u16::from_le_bytes(b.get(at..at + 2)?.try_into().ok()?) as u64)
}

fn u32_at(b: &[u8], at: usize) -> Option<u64> {
    Some(u32::from_le_bytes(b.get(at..at + 4)?.try_into().ok()?) as u64)
}

fn u64_at(b: &[u8], at: usize) -> Option<u64> {
    Some(u64::from_le_bytes(b.get(at..at + 8)?.try_into().ok()?))
}

fn zip_entries(b: &[u8]) -> Result<Vec<(String, &[u8])>, CheckpointError> {
    let bad = |what: &str| CheckpointError::Archive(what.to_string());
    let end = (0..b.len().saturating_sub(21))
        .rev()
        .take(65_557)
        .find(|&at| u32_at(b, at) == Some(0x0605_4b50))
        .ok_or_else(|| bad("no end of central directory"))?;
    let mut count = u16_at(b, end + 10).ok_or_else(|| bad("truncated"))?;
    let mut dir = u32_at(b, end + 16).ok_or_else(|| bad("truncated"))?;
    // ZIP64: the real values live in the ZIP64 end record.
    if (count == 0xFFFF || dir == 0xFFFF_FFFF)
        && end >= 20
        && u32_at(b, end - 20) == Some(0x0706_4b50)
    {
        let record = u64_at(b, end - 12).ok_or_else(|| bad("truncated"))? as usize;
        if u32_at(b, record) != Some(0x0606_4b50) {
            return Err(bad("broken ZIP64 record"));
        }
        count = u64_at(b, record + 32).ok_or_else(|| bad("truncated"))?;
        dir = u64_at(b, record + 48).ok_or_else(|| bad("truncated"))?;
    }
    let mut at = dir as usize;
    let mut entries = Vec::with_capacity(count as usize);
    for _ in 0..count {
        if u32_at(b, at) != Some(0x0201_4b50) {
            return Err(bad("broken central directory"));
        }
        let method = u16_at(b, at + 10).ok_or_else(|| bad("truncated"))?;
        let mut size = u32_at(b, at + 20).ok_or_else(|| bad("truncated"))?;
        let name_len = u16_at(b, at + 28).ok_or_else(|| bad("truncated"))? as usize;
        let extra_len = u16_at(b, at + 30).ok_or_else(|| bad("truncated"))? as usize;
        let comment_len = u16_at(b, at + 32).ok_or_else(|| bad("truncated"))? as usize;
        let mut local = u32_at(b, at + 42).ok_or_else(|| bad("truncated"))?;
        let name = b
            .get(at + 46..at + 46 + name_len)
            .and_then(|n| std::str::from_utf8(n).ok())
            .ok_or_else(|| bad("bad entry name"))?
            .to_string();
        // ZIP64 extra field: uncompressed, compressed, offset, in that
        // order, each present only when its 32-bit field is saturated.
        let extra = b
            .get(at + 46 + name_len..at + 46 + name_len + extra_len)
            .ok_or_else(|| bad("truncated"))?;
        let mut e = 0;
        while e + 4 <= extra.len() {
            let id = u16_at(extra, e).unwrap_or(0);
            let len = u16_at(extra, e + 2).unwrap_or(0) as usize;
            if id == 0x0001 {
                let mut f = e + 4;
                if size == 0xFFFF_FFFF {
                    size = u64_at(extra, f).ok_or_else(|| bad("bad ZIP64 extra"))?;
                    f += 8;
                }
                if u32_at(b, at + 20) == Some(0xFFFF_FFFF)
                    || u32_at(b, at + 24) == Some(0xFFFF_FFFF)
                {
                    f += 8;
                }
                if local == 0xFFFF_FFFF {
                    local = u64_at(extra, f).ok_or_else(|| bad("bad ZIP64 extra"))?;
                }
            }
            e += 4 + len;
        }
        if method != 0 {
            return Err(bad("compressed entry"));
        }
        let local = local as usize;
        if u32_at(b, local) != Some(0x0403_4b50) {
            return Err(bad("broken local header"));
        }
        let start = local
            + 30
            + u16_at(b, local + 26).ok_or_else(|| bad("truncated"))? as usize
            + u16_at(b, local + 28).ok_or_else(|| bad("truncated"))? as usize;
        let data = b
            .get(start..start + size as usize)
            .ok_or_else(|| bad("entry overruns the archive"))?;
        entries.push((name, data));
        at += 46 + name_len + extra_len + comment_len;
    }
    Ok(entries)
}

// ------------------------------------------------------------- pickle --

#[derive(Debug, Clone)]
enum Value {
    None,
    Bool,
    Int(i64),
    Float,
    Str(String),
    Tuple(Vec<Value>),
    Dict(Vec<(Value, Value)>),
    Global(String),
    /// A persistent id: `('storage', torch.XStorage, key, location, numel)`.
    Storage {
        key: String,
        dtype: String,
    },
    Tensor {
        storage: String,
        dtype: String,
        offset: usize,
        size: Vec<usize>,
        stride: Vec<usize>,
    },
    Mark,
}

fn dict_get<'v>(value: &'v Value, key: &str) -> Option<&'v Value> {
    let Value::Dict(items) = value else {
        return None;
    };
    items
        .iter()
        .find(|(k, _)| matches!(k, Value::Str(s) if s == key))
        .map(|(_, v)| v)
}

fn as_usizes(value: &Value) -> Option<Vec<usize>> {
    let Value::Tuple(items) = value else {
        return None;
    };
    items
        .iter()
        .map(|v| match v {
            Value::Int(i) if *i >= 0 => Some(*i as usize),
            _ => None,
        })
        .collect()
}

fn unpickle(data: &[u8]) -> Result<Value, CheckpointError> {
    let fail = |what: String| CheckpointError::Pickle(what);
    let mut stack: Vec<Value> = Vec::new();
    let mut memo: HashMap<u32, Value> = HashMap::new();
    let mut at = 0usize;
    let take = |at: &mut usize, n: usize| -> Result<&[u8], CheckpointError> {
        let slice = data
            .get(*at..*at + n)
            .ok_or_else(|| CheckpointError::Pickle("truncated".into()))?;
        *at += n;
        Ok(slice)
    };
    let pop_mark = |stack: &mut Vec<Value>| -> Result<Vec<Value>, CheckpointError> {
        let mark = stack
            .iter()
            .rposition(|v| matches!(v, Value::Mark))
            .ok_or_else(|| CheckpointError::Pickle("no mark".into()))?;
        let items = stack.split_off(mark + 1);
        stack.pop();
        Ok(items)
    };
    loop {
        let op = take(&mut at, 1)?[0];
        match op {
            0x80 => {
                let proto = take(&mut at, 1)?[0];
                if proto > 2 {
                    return Err(fail(format!("protocol {proto}")));
                }
            }
            b'}' => stack.push(Value::Dict(Vec::new())),
            b')' => stack.push(Value::Tuple(Vec::new())),
            b'(' => stack.push(Value::Mark),
            b'N' => stack.push(Value::None),
            0x88 | 0x89 => stack.push(Value::Bool),
            b'K' => stack.push(Value::Int(take(&mut at, 1)?[0] as i64)),
            b'M' => {
                let b = take(&mut at, 2)?;
                stack.push(Value::Int(u16::from_le_bytes([b[0], b[1]]) as i64));
            }
            b'J' => {
                let b = take(&mut at, 4)?;
                stack.push(Value::Int(
                    i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as i64
                ));
            }
            b'G' => {
                take(&mut at, 8)?;
                stack.push(Value::Float);
            }
            b'X' => {
                let b = take(&mut at, 4)?;
                let n = u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize;
                let s =
                    std::str::from_utf8(take(&mut at, n)?).map_err(|_| fail("bad utf-8".into()))?;
                stack.push(Value::Str(s.to_string()));
            }
            b'c' => {
                let rest = &data[at..];
                let first = rest
                    .iter()
                    .position(|&c| c == b'\n')
                    .ok_or_else(|| fail("bad global".into()))?;
                let second = rest[first + 1..]
                    .iter()
                    .position(|&c| c == b'\n')
                    .ok_or_else(|| fail("bad global".into()))?;
                let module = String::from_utf8_lossy(&rest[..first]);
                let name = String::from_utf8_lossy(&rest[first + 1..first + 1 + second]);
                at += first + second + 2;
                let global = format!("{module}.{name}");
                match global.as_str() {
                    "collections.OrderedDict"
                    | "torch._utils._rebuild_tensor_v2"
                    | "torch.FloatStorage"
                    | "torch.LongStorage" => stack.push(Value::Global(global)),
                    other => return Err(fail(format!("global {other}"))),
                }
            }
            b'q' => {
                let key = take(&mut at, 1)?[0] as u32;
                memo.insert(
                    key,
                    stack.last().cloned().ok_or_else(|| fail("empty".into()))?,
                );
            }
            b'r' => {
                let b = take(&mut at, 4)?;
                let key = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                memo.insert(
                    key,
                    stack.last().cloned().ok_or_else(|| fail("empty".into()))?,
                );
            }
            b'h' => {
                let key = take(&mut at, 1)?[0] as u32;
                stack.push(memo.get(&key).cloned().ok_or_else(|| fail("memo".into()))?);
            }
            b'j' => {
                let b = take(&mut at, 4)?;
                let key = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                stack.push(memo.get(&key).cloned().ok_or_else(|| fail("memo".into()))?);
            }
            b't' => {
                let items = pop_mark(&mut stack)?;
                stack.push(Value::Tuple(items));
            }
            0x85..=0x87 => {
                let n = (op - 0x84) as usize;
                if stack.len() < n {
                    return Err(fail("short stack".into()));
                }
                let items = stack.split_off(stack.len() - n);
                stack.push(Value::Tuple(items));
            }
            b'Q' => {
                let pid = stack.pop().ok_or_else(|| fail("empty".into()))?;
                let Value::Tuple(parts) = pid else {
                    return Err(fail("persistent id".into()));
                };
                match parts.as_slice() {
                    [
                        Value::Str(kind),
                        Value::Global(class),
                        Value::Str(key),
                        _,
                        _,
                    ] if kind == "storage" => stack.push(Value::Storage {
                        key: key.clone(),
                        dtype: class.trim_start_matches("torch.").to_string(),
                    }),
                    _ => return Err(fail("persistent id".into())),
                }
            }
            b'R' => {
                let args = stack.pop().ok_or_else(|| fail("empty".into()))?;
                let func = stack.pop().ok_or_else(|| fail("empty".into()))?;
                let Value::Global(func) = func else {
                    return Err(fail("reduce on a non-global".into()));
                };
                let Value::Tuple(args) = args else {
                    return Err(fail("reduce arguments".into()));
                };
                match func.as_str() {
                    "collections.OrderedDict" => stack.push(Value::Dict(Vec::new())),
                    "torch._utils._rebuild_tensor_v2" => {
                        let (Some(Value::Storage { key, dtype }), Some(Value::Int(offset))) =
                            (args.first(), args.get(1))
                        else {
                            return Err(fail("tensor arguments".into()));
                        };
                        let size = args.get(2).and_then(as_usizes);
                        let stride = args.get(3).and_then(as_usizes);
                        let (Some(size), Some(stride)) = (size, stride) else {
                            return Err(fail("tensor shape".into()));
                        };
                        stack.push(Value::Tensor {
                            storage: key.clone(),
                            dtype: dtype.clone(),
                            offset: *offset as usize,
                            size,
                            stride,
                        });
                    }
                    other => return Err(fail(format!("call {other}"))),
                }
            }
            b's' | b'u' => {
                let items = if op == b'u' {
                    pop_mark(&mut stack)?
                } else {
                    let v = stack.pop().ok_or_else(|| fail("empty".into()))?;
                    let k = stack.pop().ok_or_else(|| fail("empty".into()))?;
                    vec![k, v]
                };
                let Some(Value::Dict(dict)) = stack.last_mut() else {
                    return Err(fail("setitems on a non-dict".into()));
                };
                let mut items = items.into_iter();
                while let (Some(k), Some(v)) = (items.next(), items.next()) {
                    dict.push((k, v));
                }
            }
            b'.' => return stack.pop().ok_or_else(|| fail("empty".into())),
            other => return Err(fail(format!("opcode 0x{other:02x}"))),
        }
    }
}
