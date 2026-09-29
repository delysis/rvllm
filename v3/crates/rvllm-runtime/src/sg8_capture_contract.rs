//! Host-only validation for a future SG8 same-input tensor capture.
//!
//! This module does not read a Metal buffer, encode a blit, or establish that
//! the production route produced any supplied bytes. A separate reviewed
//! transport and live dispatch gate are prerequisites for device evidence.
#![forbid(unsafe_code)]

use crate::kernel_game::parse_strict_json;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

pub const SCHEMA: &str = "rvllm.sg8-safe-capture-contract.v1";
pub const MAX_PAYLOAD_BYTES: u64 = 512 * 1024 * 1024;
pub const MAX_METADATA_BYTES: usize = 2 * 1024 * 1024;
const FIRST_LAYER: u8 = 4;
const LAST_LAYER: u8 = 6;
const FUSED_UNAVAILABLE: &str = "fused_interior_not_materialized";
const PROJECTION: &str = "research_donor12b_sg8_native_projection";
const GATE: &str = "research_donor12b_sg8_native_gate";
const LOCAL: &str = "research_donor12b_sg8_local_attention";
const GLOBAL: &str = "research_donor12b_sg8_global_attention";

type Check<T> = Result<T, String>;

fn require(ok: bool, message: &str) -> Check<()> {
    if ok {
        Ok(())
    } else {
        Err(message.to_owned())
    }
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Identity {
    pub model_sha256: String,
    pub config_sha256: String,
    pub tokenizer_sha256: String,
    pub metallib_sha256: String,
    pub executable_sha256: String,
    pub source_sha256: String,
}

impl Identity {
    fn validate(&self) -> Check<()> {
        for value in [
            &self.model_sha256,
            &self.config_sha256,
            &self.tokenizer_sha256,
            &self.metallib_sha256,
            &self.executable_sha256,
            &self.source_sha256,
        ] {
            require(valid_sha256(value), "malformed source identity")?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Route {
    Off,
    Sg8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Boundary {
    InputResidual,
    InputNorm,
    PackedQkv,
    QPostRope,
    KPostRope,
    VNormalized,
    AttendedK,
    AttendedV,
    AttentionOutput,
    AttentionResidual,
    GateUp,
    FfnBranch,
    OutputResidual,
    FinalResidual,
}

impl Boundary {
    fn may_be_unavailable(self) -> bool {
        matches!(self, Self::PackedQkv | Self::GateUp)
    }
}

const LAYER_BOUNDARIES: [Boundary; 13] = [
    Boundary::InputResidual,
    Boundary::InputNorm,
    Boundary::PackedQkv,
    Boundary::QPostRope,
    Boundary::KPostRope,
    Boundary::VNormalized,
    Boundary::AttendedK,
    Boundary::AttendedV,
    Boundary::AttentionOutput,
    Boundary::AttentionResidual,
    Boundary::GateUp,
    Boundary::FfnBranch,
    Boundary::OutputResidual,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Dtype {
    Bf16Le,
    F32Le,
}

impl Dtype {
    fn width(self) -> u64 {
        match self {
            Self::Bf16Le => 2,
            Self::F32Le => 4,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Availability {
    Materialized,
    Unavailable,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Point {
    pub layer: Option<u8>,
    pub boundary: Boundary,
    pub shape: Vec<u64>,
    pub dtype: Dtype,
    pub availability: Availability,
    pub dispatch_prefix: BTreeMap<String, u64>,
}

impl Point {
    fn byte_len(&self) -> Check<u64> {
        require((1..=4).contains(&self.shape.len()), "invalid tensor rank")?;
        let elements = self.shape.iter().try_fold(1_u64, |count, &dim| {
            require(dim > 0, "zero tensor dimension")?;
            count
                .checked_mul(dim)
                .ok_or("tensor shape overflow".to_owned())
        })?;
        elements
            .checked_mul(self.dtype.width())
            .filter(|&bytes| bytes <= MAX_PAYLOAD_BYTES)
            .ok_or("tensor byte length exceeds cap".to_owned())
    }

    fn validate(&self, route: Route, kv: Option<&LogicalKv>) -> Check<()> {
        self.byte_len()?;
        let (expected_shape, expected_dtype) = expected_descriptor(self.boundary, kv)?;
        require(
            self.shape == expected_shape && self.dtype == expected_dtype,
            "capture tensor geometry or dtype mismatch",
        )?;
        if self.availability == Availability::Unavailable {
            require(
                self.boundary.may_be_unavailable(),
                "required boundary unavailable",
            )?;
        }
        for (name, &count) in &self.dispatch_prefix {
            require(count > 0, "zero dispatch count")?;
            require(
                [PROJECTION, GATE, LOCAL, GLOBAL].contains(&name.as_str()),
                "unknown SG8 dispatch family",
            )?;
        }
        if route == Route::Off {
            require(
                self.dispatch_prefix.is_empty(),
                "off route has SG8 dispatch",
            )?;
        }
        Ok(())
    }
}

fn expected_descriptor(boundary: Boundary, kv: Option<&LogicalKv>) -> Check<(Vec<u64>, Dtype)> {
    let result = match boundary {
        Boundary::InputResidual
        | Boundary::InputNorm
        | Boundary::AttentionResidual
        | Boundary::FfnBranch
        | Boundary::OutputResidual
        | Boundary::FinalResidual => (vec![1, 3840], Dtype::Bf16Le),
        Boundary::GateUp => (vec![1, 30720], Dtype::F32Le),
        other => {
            let kv = kv.ok_or("missing layer geometry")?;
            match other {
                Boundary::PackedQkv => {
                    (vec![1, (16 + 2 * kv.kv_heads) * kv.head_dim], Dtype::F32Le)
                }
                Boundary::QPostRope | Boundary::AttentionOutput => {
                    (vec![1, 16, kv.head_dim], Dtype::Bf16Le)
                }
                Boundary::KPostRope | Boundary::VNormalized => {
                    (vec![1, kv.kv_heads, kv.head_dim], Dtype::Bf16Le)
                }
                Boundary::AttendedK | Boundary::AttendedV => (
                    vec![kv.valid_tokens, kv.kv_heads, kv.head_dim],
                    Dtype::Bf16Le,
                ),
                _ => return Err("unsupported layer boundary".into()),
            }
        }
    };
    Ok(result)
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LogicalKv {
    pub consumer_layer: u8,
    pub producer_layer: u8,
    pub kv_heads: u64,
    pub head_dim: u64,
    pub page_tokens: u64,
    pub valid_tokens: u64,
    pub physical_page_count: u64,
    pub block_table: Vec<u64>,
}

impl LogicalKv {
    fn validate(&self, context: u64) -> Check<()> {
        require(
            (FIRST_LAYER..=LAST_LAYER).contains(&self.consumer_layer)
                && self.producer_layer <= self.consumer_layer,
            "invalid logical KV producer alias",
        )?;
        let expected_geometry = if self.consumer_layer == 5 {
            (1, 512)
        } else {
            (8, 256)
        };
        require(
            (self.kv_heads, self.head_dim) == expected_geometry
                && self.page_tokens.is_power_of_two()
                && self.page_tokens <= 1024
                && self.valid_tokens == context
                && self.physical_page_count > 0,
            "invalid logical KV geometry",
        )?;
        require(
            self.page_tokens > 0 && self.physical_page_count >= self.block_table.len() as u64,
            "invalid logical KV geometry",
        )?;
        let expected_pages = self
            .valid_tokens
            .checked_add(self.page_tokens - 1)
            .ok_or("logical KV page count overflow")?
            / self.page_tokens;
        require(
            u64::try_from(self.block_table.len()).ok() == Some(expected_pages),
            "logical KV page table length mismatch",
        )?;
        let mut seen = std::collections::BTreeSet::new();
        for &page in &self.block_table {
            require(
                page < self.physical_page_count && seen.insert(page),
                "invalid or duplicate physical KV page",
            )?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Plan {
    pub schema: String,
    pub identity: Identity,
    pub route: Route,
    pub ticket_id: u64,
    pub prompt_tokens: u64,
    pub decode_ordinal: u64,
    pub execution_slot: u32,
    pub position: u64,
    pub context: u64,
    pub conditioning_token_ids: Vec<u32>,
    pub input_ids_sha256: String,
    pub max_payload_bytes: u64,
    pub logical_kv: Vec<LogicalKv>,
    pub points: Vec<Point>,
}

impl Plan {
    pub fn validate(&self) -> Check<()> {
        require(self.schema == SCHEMA, "wrong capture schema")?;
        self.identity.validate()?;
        require(
            valid_sha256(&self.input_ids_sha256),
            "malformed input identity",
        )?;
        require(
            self.ticket_id > 0
                && matches!(self.prompt_tokens, 256 | 512)
                && (1..=64).contains(&self.decode_ordinal)
                && self.execution_slot == 0,
            "wrong ticket, prompt length, decode ordinal or live slot",
        )?;
        require(
            self.prompt_tokens.checked_add(self.decode_ordinal - 1) == Some(self.context)
                && self.position.checked_add(1) == Some(self.context),
            "inconsistent position and context",
        )?;
        require(
            u64::try_from(self.conditioning_token_ids.len()).ok() == Some(self.context)
                && self.conditioning_token_ids.iter().all(|&id| id < 262_144)
                && self.input_ids_sha256
                    == digest(
                        &serde_json::to_vec(&self.conditioning_token_ids)
                            .map_err(|error| error.to_string())?,
                    ),
            "conditioning token history mismatch",
        )?;
        require(
            (1..=MAX_PAYLOAD_BYTES).contains(&self.max_payload_bytes),
            "invalid payload budget",
        )?;
        require(
            self.logical_kv.len() == 3 && self.points.len() == 3 * LAYER_BOUNDARIES.len() + 1,
            "incomplete capture plan",
        )?;
        for (i, kv) in self.logical_kv.iter().enumerate() {
            require(
                kv.consumer_layer == FIRST_LAYER + i as u8,
                "logical KV consumer order mismatch",
            )?;
            kv.validate(self.context)?;
        }
        for (i, point) in self.points.iter().enumerate() {
            let expected = if i == self.points.len() - 1 {
                (None, Boundary::FinalResidual)
            } else {
                (
                    Some(FIRST_LAYER + (i / LAYER_BOUNDARIES.len()) as u8),
                    LAYER_BOUNDARIES[i % LAYER_BOUNDARIES.len()],
                )
            };
            require(
                (point.layer, point.boundary) == expected,
                "capture points out of order or duplicated",
            )?;
            let kv = point
                .layer
                .map(|layer| &self.logical_kv[usize::from(layer - FIRST_LAYER)]);
            point.validate(self.route, kv)?;
        }
        if self.route == Route::Sg8 {
            for layer_points in
                self.points[..self.points.len() - 1].chunks_exact(LAYER_BOUNDARIES.len())
            {
                for pair in layer_points.windows(2) {
                    for name in [PROJECTION, GATE, LOCAL, GLOBAL] {
                        require(
                            pair[1].dispatch_prefix.get(name).copied().unwrap_or(0)
                                >= pair[0].dispatch_prefix.get(name).copied().unwrap_or(0),
                            "SG8 dispatch prefix regressed within layer",
                        )?;
                    }
                }
                let input = &layer_points[0].dispatch_prefix;
                let q = &layer_points[3].dispatch_prefix;
                let before_attention = &layer_points[7].dispatch_prefix;
                let attention = &layer_points[8].dispatch_prefix;
                let before_gate = &layer_points[9].dispatch_prefix;
                let output = &layer_points[12].dispatch_prefix;
                require(
                    q.get(PROJECTION).copied().unwrap_or(0)
                        > input.get(PROJECTION).copied().unwrap_or(0),
                    "SG8 Q boundary lacks named projection dispatch",
                )?;
                let local = attention.get(LOCAL).copied().unwrap_or(0)
                    - before_attention.get(LOCAL).copied().unwrap_or(0);
                let global = attention.get(GLOBAL).copied().unwrap_or(0)
                    - before_attention.get(GLOBAL).copied().unwrap_or(0);
                let expected_attention = if layer_points[0].layer == Some(5) {
                    (0, 1)
                } else {
                    (1, 0)
                };
                require(
                    (local, global) == expected_attention,
                    "SG8 attention boundary lacks the layer's named attention dispatch",
                )?;
                require(
                    output.get(PROJECTION).copied().unwrap_or(0)
                        > q.get(PROJECTION).copied().unwrap_or(0)
                        && output.get(GATE).copied().unwrap_or(0)
                            > before_gate.get(GATE).copied().unwrap_or(0),
                    "SG8 output lacks named projection and gate dispatch",
                )?;
            }
        }
        let mut bytes = 0_u64;
        for point in &self.points {
            if point.availability == Availability::Materialized {
                bytes = bytes
                    .checked_add(point.byte_len()?)
                    .ok_or("payload budget overflow")?;
            }
        }
        require(
            bytes <= self.max_payload_bytes,
            "capture exceeds payload budget",
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Storage {
    Stored {
        offset: u64,
        len: u64,
        sha256: String,
    },
    Unavailable {
        reason: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Record {
    pub layer: Option<u8>,
    pub boundary: Boundary,
    pub dispatch_prefix: BTreeMap<String, u64>,
    pub storage: Storage,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GpuStatus {
    Completed,
    Error,
    NotCompleted,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Receipt {
    pub schema: String,
    pub plan_sha256: String,
    pub ticket_id: u64,
    pub gpu_status: GpuStatus,
    pub execution_slot: u32,
    pub position: u64,
    pub context: u64,
    pub input_ids_sha256: String,
    pub logical_kv: Vec<LogicalKv>,
    pub payload_sha256: String,
    pub records: Vec<Record>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedCapture {
    pub stored_points: usize,
    pub payload_bytes: u64,
    pub payload_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct StoredDifference {
    pub layer: Option<u8>,
    pub boundary: Boundary,
    pub element_index: u64,
    pub off_bits: u32,
    pub sg8_bits: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct BoundaryDifference {
    pub layer: Option<u8>,
    pub boundary: Boundary,
    pub differing_elements: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct PairComparison {
    pub compared_points: usize,
    pub compared_elements: u64,
    pub differing_elements: u64,
    pub first_stored_difference: Option<StoredDifference>,
    pub differing_boundaries: Vec<BoundaryDifference>,
}

/// Validate supplied evidence only. This cannot establish the provenance of
/// bytes before the future platform transport hands them to this function.
pub fn verify(plan_json: &[u8], receipt_json: &[u8], payload: &[u8]) -> Check<VerifiedCapture> {
    require(
        plan_json.len() <= MAX_METADATA_BYTES && receipt_json.len() <= MAX_METADATA_BYTES,
        "capture metadata exceeds byte limit",
    )?;
    let plan: Plan = parse_strict_json(plan_json).map_err(|e| e.to_string())?;
    plan.validate()?;
    let receipt: Receipt = parse_strict_json(receipt_json).map_err(|e| e.to_string())?;
    require(receipt.schema == SCHEMA, "wrong receipt schema")?;
    require(
        receipt.plan_sha256 == digest(plan_json),
        "receipt is not bound to exact plan bytes",
    )?;
    require(
        receipt.ticket_id == plan.ticket_id && receipt.gpu_status == GpuStatus::Completed,
        "wrong ticket or incomplete GPU command",
    )?;
    require(
        receipt.execution_slot == plan.execution_slot
            && receipt.position == plan.position
            && receipt.context == plan.context
            && receipt.input_ids_sha256 == plan.input_ids_sha256
            && receipt.logical_kv == plan.logical_kv,
        "capture history or logical KV mismatch",
    )?;
    require(
        receipt.records.len() == plan.points.len(),
        "missing or extra capture record",
    )?;
    let payload_bytes = u64::try_from(payload.len()).map_err(|_| "payload length overflow")?;
    require(
        payload_bytes <= plan.max_payload_bytes && receipt.payload_sha256 == digest(payload),
        "payload budget or full hash mismatch",
    )?;
    let mut cursor = 0_u64;
    let mut stored_points = 0;
    for (point, record) in plan.points.iter().zip(&receipt.records) {
        require(
            record.layer == point.layer
                && record.boundary == point.boundary
                && record.dispatch_prefix == point.dispatch_prefix,
            "capture descriptor or dispatch prefix mismatch",
        )?;
        match (&point.availability, &record.storage) {
            (
                Availability::Materialized,
                Storage::Stored {
                    offset,
                    len,
                    sha256,
                },
            ) => {
                require(
                    *offset == cursor && *len == point.byte_len()?,
                    "capture span is missing, overlapping or out of order",
                )?;
                cursor = cursor.checked_add(*len).ok_or("capture span overflow")?;
                require(cursor <= payload_bytes, "capture span exceeds payload")?;
                let start = usize::try_from(*offset).map_err(|_| "capture offset overflow")?;
                let end = usize::try_from(cursor).map_err(|_| "capture end overflow")?;
                require(
                    valid_sha256(sha256) && *sha256 == digest(&payload[start..end]),
                    "tensor hash mismatch",
                )?;
                stored_points += 1;
            }
            (Availability::Unavailable, Storage::Unavailable { reason }) => {
                require(reason == FUSED_UNAVAILABLE, "unapproved unavailable reason")?;
            }
            _ => return Err("capture availability differs from plan".into()),
        }
    }
    require(cursor == payload_bytes, "undeclared trailing payload bytes")?;
    Ok(VerifiedCapture {
        stored_points,
        payload_bytes,
        payload_sha256: receipt.payload_sha256,
    })
}

fn stored_bytes<'a>(record: &Record, payload: &'a [u8]) -> Check<Option<&'a [u8]>> {
    let Storage::Stored { offset, len, .. } = &record.storage else {
        return Ok(None);
    };
    let start = usize::try_from(*offset).map_err(|_| "capture offset overflow")?;
    let end = offset
        .checked_add(*len)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or("capture end overflow")?;
    payload
        .get(start..end)
        .map(Some)
        .ok_or("capture span exceeds payload".into())
}

/// Compare two independently supplied captures after validating each one.
/// The first difference is in descriptor order, not proof of the first
/// arithmetic cause or of either payload's live GPU provenance.
pub fn compare_pair(
    off_plan_json: &[u8],
    off_receipt_json: &[u8],
    off_payload: &[u8],
    sg8_plan_json: &[u8],
    sg8_receipt_json: &[u8],
    sg8_payload: &[u8],
) -> Check<PairComparison> {
    verify(off_plan_json, off_receipt_json, off_payload)?;
    verify(sg8_plan_json, sg8_receipt_json, sg8_payload)?;
    let off: Plan = parse_strict_json(off_plan_json).map_err(|e| e.to_string())?;
    let sg8: Plan = parse_strict_json(sg8_plan_json).map_err(|e| e.to_string())?;
    let off_receipt: Receipt = parse_strict_json(off_receipt_json).map_err(|e| e.to_string())?;
    let sg8_receipt: Receipt = parse_strict_json(sg8_receipt_json).map_err(|e| e.to_string())?;
    require(
        off.route == Route::Off && sg8.route == Route::Sg8,
        "paired capture routes are not selector-off then SG8",
    )?;
    require(
        off.identity == sg8.identity
            && off.prompt_tokens == sg8.prompt_tokens
            && off.decode_ordinal == sg8.decode_ordinal
            && off.execution_slot == sg8.execution_slot
            && off.position == sg8.position
            && off.context == sg8.context
            && off.conditioning_token_ids == sg8.conditioning_token_ids
            && off.input_ids_sha256 == sg8.input_ids_sha256
            && off.logical_kv == sg8.logical_kv,
        "paired capture identities, input histories or logical KV maps differ",
    )?;
    let mut result = PairComparison {
        compared_points: 0,
        compared_elements: 0,
        differing_elements: 0,
        first_stored_difference: None,
        differing_boundaries: Vec::new(),
    };
    for (((off_point, sg8_point), off_record), sg8_record) in off
        .points
        .iter()
        .zip(&sg8.points)
        .zip(&off_receipt.records)
        .zip(&sg8_receipt.records)
    {
        require(
            off_point.layer == sg8_point.layer
                && off_point.boundary == sg8_point.boundary
                && off_point.shape == sg8_point.shape
                && off_point.dtype == sg8_point.dtype
                && off_point.availability == sg8_point.availability,
            "paired capture descriptors or availability differ",
        )?;
        let (Some(off_bytes), Some(sg8_bytes)) = (
            stored_bytes(off_record, off_payload)?,
            stored_bytes(sg8_record, sg8_payload)?,
        ) else {
            continue;
        };
        require(
            off_bytes.len() == sg8_bytes.len(),
            "paired tensor lengths differ",
        )?;
        result.compared_points += 1;
        let width = usize::try_from(off_point.dtype.width()).map_err(|_| "invalid dtype width")?;
        let mut boundary_differences = 0_u64;
        for (index, (left, right)) in off_bytes
            .chunks_exact(width)
            .zip(sg8_bytes.chunks_exact(width))
            .enumerate()
        {
            result.compared_elements = result
                .compared_elements
                .checked_add(1)
                .ok_or("comparison element count overflow")?;
            if left != right {
                let bits = |bytes: &[u8]| -> u32 {
                    if width == 2 {
                        u32::from(u16::from_le_bytes([bytes[0], bytes[1]]))
                    } else {
                        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
                    }
                };
                result
                    .first_stored_difference
                    .get_or_insert(StoredDifference {
                        layer: off_point.layer,
                        boundary: off_point.boundary,
                        element_index: u64::try_from(index)
                            .map_err(|_| "comparison element index overflow")?,
                        off_bits: bits(left),
                        sg8_bits: bits(right),
                    });
                boundary_differences = boundary_differences
                    .checked_add(1)
                    .ok_or("boundary difference count overflow")?;
            }
        }
        if boundary_differences > 0 {
            result.differing_elements = result
                .differing_elements
                .checked_add(boundary_differences)
                .ok_or("comparison difference count overflow")?;
            result.differing_boundaries.push(BoundaryDifference {
                layer: off_point.layer,
                boundary: off_point.boundary,
                differing_elements: boundary_differences,
            });
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (Plan, Receipt, Vec<u8>) {
        let h = digest(b"pinned");
        let identity = Identity {
            model_sha256: h.clone(),
            config_sha256: h.clone(),
            tokenizer_sha256: h.clone(),
            metallib_sha256: h.clone(),
            executable_sha256: h.clone(),
            source_sha256: h.clone(),
        };
        let logical_kv = (FIRST_LAYER..=LAST_LAYER)
            .map(|layer| LogicalKv {
                consumer_layer: layer,
                producer_layer: layer,
                kv_heads: if layer == 5 { 1 } else { 8 },
                head_dim: if layer == 5 { 512 } else { 256 },
                page_tokens: 16,
                valid_tokens: 256,
                physical_page_count: 16,
                block_table: (0..16).collect(),
            })
            .collect::<Vec<_>>();
        let mut points = Vec::new();
        for layer in FIRST_LAYER..=LAST_LAYER {
            for boundary in LAYER_BOUNDARIES {
                let (shape, dtype) = expected_descriptor(
                    boundary,
                    Some(&logical_kv[usize::from(layer - FIRST_LAYER)]),
                )
                .unwrap();
                points.push(Point {
                    layer: Some(layer),
                    boundary,
                    shape,
                    dtype,
                    availability: if boundary.may_be_unavailable() {
                        Availability::Unavailable
                    } else {
                        Availability::Materialized
                    },
                    dispatch_prefix: BTreeMap::new(),
                });
            }
        }
        points.push(Point {
            layer: None,
            boundary: Boundary::FinalResidual,
            shape: vec![1, 3840],
            dtype: Dtype::Bf16Le,
            availability: Availability::Materialized,
            dispatch_prefix: BTreeMap::new(),
        });
        let conditioning_token_ids = vec![42; 256];
        let plan = Plan {
            schema: SCHEMA.into(),
            identity,
            route: Route::Off,
            ticket_id: 7,
            prompt_tokens: 256,
            decode_ordinal: 1,
            execution_slot: 0,
            position: 255,
            context: 256,
            input_ids_sha256: digest(&serde_json::to_vec(&conditioning_token_ids).unwrap()),
            conditioning_token_ids,
            max_payload_bytes: 16 * 1024 * 1024,
            logical_kv: logical_kv.clone(),
            points,
        };
        let mut payload = Vec::new();
        let mut records = Vec::new();
        for point in &plan.points {
            let storage = if point.availability == Availability::Materialized {
                let offset = payload.len() as u64;
                payload.resize(
                    payload.len() + usize::try_from(point.byte_len().unwrap()).unwrap(),
                    point.boundary as u8,
                );
                Storage::Stored {
                    offset,
                    len: point.byte_len().unwrap(),
                    sha256: digest(&payload[offset as usize..]),
                }
            } else {
                Storage::Unavailable {
                    reason: FUSED_UNAVAILABLE.into(),
                }
            };
            records.push(Record {
                layer: point.layer,
                boundary: point.boundary,
                dispatch_prefix: BTreeMap::new(),
                storage,
            });
        }
        let receipt = Receipt {
            schema: SCHEMA.into(),
            plan_sha256: digest(&serde_json::to_vec(&plan).unwrap()),
            ticket_id: plan.ticket_id,
            gpu_status: GpuStatus::Completed,
            execution_slot: 0,
            position: 255,
            context: 256,
            input_ids_sha256: plan.input_ids_sha256.clone(),
            logical_kv,
            payload_sha256: digest(&payload),
            records,
        };
        (plan, receipt, payload)
    }

    fn check(plan: &Plan, receipt: &Receipt, payload: &[u8]) -> Check<VerifiedCapture> {
        verify(
            &serde_json::to_vec(plan).unwrap(),
            &serde_json::to_vec(receipt).unwrap(),
            payload,
        )
    }

    fn sg8_fixture() -> (Plan, Receipt, Vec<u8>) {
        let (mut plan, mut receipt, payload) = fixture();
        plan.route = Route::Sg8;
        for layer in 0..3 {
            let base = layer * LAYER_BOUNDARIES.len();
            let attention_name = if layer == 1 { GLOBAL } else { LOCAL };
            for i in 3..LAYER_BOUNDARIES.len() {
                plan.points[base + i]
                    .dispatch_prefix
                    .insert(PROJECTION.into(), if i == 12 { 2 } else { 1 });
            }
            for i in 8..LAYER_BOUNDARIES.len() {
                plan.points[base + i]
                    .dispatch_prefix
                    .insert(attention_name.into(), 1);
            }
            for i in 11..LAYER_BOUNDARIES.len() {
                plan.points[base + i].dispatch_prefix.insert(GATE.into(), 1);
            }
        }
        for (point, record) in plan.points.iter().zip(&mut receipt.records) {
            record.dispatch_prefix = point.dispatch_prefix.clone();
        }
        receipt.plan_sha256 = digest(&serde_json::to_vec(&plan).unwrap());
        (plan, receipt, payload)
    }

    fn paired(
        off: &(Plan, Receipt, Vec<u8>),
        sg8: &(Plan, Receipt, Vec<u8>),
    ) -> Check<PairComparison> {
        compare_pair(
            &serde_json::to_vec(&off.0).unwrap(),
            &serde_json::to_vec(&off.1).unwrap(),
            &off.2,
            &serde_json::to_vec(&sg8.0).unwrap(),
            &serde_json::to_vec(&sg8.1).unwrap(),
            &sg8.2,
        )
    }

    #[test]
    fn complete_owned_payload_is_accepted() {
        let (plan, receipt, payload) = fixture();
        assert_eq!(
            check(&plan, &receipt, &payload).unwrap().payload_bytes,
            payload.len() as u64
        );
    }

    #[test]
    fn wrong_ticket_status_and_history_fail() {
        let (plan, mut receipt, payload) = fixture();
        receipt.ticket_id += 1;
        assert!(check(&plan, &receipt, &payload).is_err());
        receipt.ticket_id = plan.ticket_id;
        receipt.gpu_status = GpuStatus::Error;
        assert!(check(&plan, &receipt, &payload).is_err());
        receipt.gpu_status = GpuStatus::Completed;
        receipt.logical_kv[0].block_table[0] = 2;
        assert!(check(&plan, &receipt, &payload).is_err());
    }

    #[test]
    fn missing_duplicate_and_reordered_points_fail() {
        let (plan, mut receipt, payload) = fixture();
        receipt.records.pop();
        assert!(check(&plan, &receipt, &payload).is_err());
        receipt.records.push(receipt.records[0].clone());
        assert!(check(&plan, &receipt, &payload).is_err());
        let (plan, mut receipt, payload) = fixture();
        receipt.records.swap(0, 1);
        assert!(check(&plan, &receipt, &payload).is_err());
    }

    #[test]
    fn altered_bytes_and_bad_spans_fail() {
        let (plan, receipt, mut payload) = fixture();
        payload[0] ^= 1;
        assert!(check(&plan, &receipt, &payload).is_err());
        let (plan, mut receipt, payload) = fixture();
        if let Storage::Stored { offset, .. } = &mut receipt.records[1].storage {
            *offset = 0;
        }
        assert!(check(&plan, &receipt, &payload).is_err());
        let (plan, receipt, mut payload) = fixture();
        payload.extend_from_slice(&[0, 0]);
        assert!(check(&plan, &receipt, &payload).is_err());
    }

    #[test]
    fn unavailable_required_point_and_bad_kv_table_fail() {
        let (mut plan, mut receipt, payload) = fixture();
        plan.points[0].availability = Availability::Unavailable;
        receipt.plan_sha256 = digest(&serde_json::to_vec(&plan).unwrap());
        assert!(check(&plan, &receipt, &payload).is_err());
        let (mut plan, mut receipt, payload) = fixture();
        plan.logical_kv[0].block_table[1] = 0;
        receipt.plan_sha256 = digest(&serde_json::to_vec(&plan).unwrap());
        assert!(check(&plan, &receipt, &payload).is_err());
    }

    #[test]
    fn unknown_fields_and_duplicate_json_keys_fail() {
        let (plan, receipt, payload) = fixture();
        let plan_json = serde_json::to_vec(&plan).unwrap();
        let receipt_json = serde_json::to_vec(&receipt).unwrap();
        let mut value: serde_json::Value = serde_json::from_slice(&receipt_json).unwrap();
        value["unexpected"] = true.into();
        assert!(verify(&plan_json, &serde_json::to_vec(&value).unwrap(), &payload).is_err());
        let duplicate = String::from_utf8(plan_json).unwrap().replacen(
            "\"schema\":",
            "\"schema\":\"wrong\",\"schema\":",
            1,
        );
        assert!(verify(duplicate.as_bytes(), &receipt_json, &payload).is_err());
    }

    #[test]
    fn dispatch_prefix_and_route_disagree_fail() {
        let (mut plan, mut receipt, payload) = fixture();
        plan.points[0].dispatch_prefix.insert(LOCAL.into(), 1);
        receipt.records[0].dispatch_prefix.insert(LOCAL.into(), 1);
        receipt.plan_sha256 = digest(&serde_json::to_vec(&plan).unwrap());
        assert!(check(&plan, &receipt, &payload).is_err());
        let (plan, mut receipt, payload) = fixture();
        receipt.records[0].dispatch_prefix.insert(LOCAL.into(), 1);
        assert!(check(&plan, &receipt, &payload).is_err());
    }

    #[test]
    fn wrong_geometry_history_identity_and_metadata_budget_fail() {
        let (mut plan, mut receipt, payload) = fixture();
        plan.points[3].shape = vec![1, 15, 256];
        receipt.plan_sha256 = digest(&serde_json::to_vec(&plan).unwrap());
        assert!(check(&plan, &receipt, &payload).is_err());

        let (mut plan, mut receipt, payload) = fixture();
        plan.conditioning_token_ids[0] += 1;
        receipt.plan_sha256 = digest(&serde_json::to_vec(&plan).unwrap());
        assert!(check(&plan, &receipt, &payload).is_err());

        let (mut plan, mut receipt, payload) = fixture();
        plan.identity.model_sha256 = "not-a-sha256".into();
        receipt.plan_sha256 = digest(&serde_json::to_vec(&plan).unwrap());
        assert!(check(&plan, &receipt, &payload).is_err());

        let (plan, receipt, payload) = fixture();
        let mut metadata = serde_json::to_vec(&plan).unwrap();
        metadata.resize(MAX_METADATA_BYTES + 1, b' ');
        assert!(verify(&metadata, &serde_json::to_vec(&receipt).unwrap(), &payload).is_err());
    }

    #[test]
    fn sg8_needs_ordered_named_dispatch_at_each_layer() {
        let (mut plan, mut receipt, payload) = fixture();
        plan.route = Route::Sg8;
        receipt.plan_sha256 = digest(&serde_json::to_vec(&plan).unwrap());
        assert!(check(&plan, &receipt, &payload).is_err());
        for layer in 0..3 {
            let base = layer * LAYER_BOUNDARIES.len();
            for i in 3..LAYER_BOUNDARIES.len() {
                plan.points[base + i]
                    .dispatch_prefix
                    .insert(PROJECTION.into(), 1);
                receipt.records[base + i]
                    .dispatch_prefix
                    .insert(PROJECTION.into(), 1);
            }
            for i in 8..LAYER_BOUNDARIES.len() {
                let attention_name = if layer == 1 { GLOBAL } else { LOCAL };
                plan.points[base + i]
                    .dispatch_prefix
                    .insert(attention_name.into(), 1);
                receipt.records[base + i]
                    .dispatch_prefix
                    .insert(attention_name.into(), 1);
            }
            for i in 11..LAYER_BOUNDARIES.len() {
                plan.points[base + i].dispatch_prefix.insert(GATE.into(), 1);
                receipt.records[base + i]
                    .dispatch_prefix
                    .insert(GATE.into(), 1);
            }
            plan.points[base + 12]
                .dispatch_prefix
                .insert(PROJECTION.into(), 2);
            receipt.records[base + 12]
                .dispatch_prefix
                .insert(PROJECTION.into(), 2);
        }
        receipt.plan_sha256 = digest(&serde_json::to_vec(&plan).unwrap());
        assert!(check(&plan, &receipt, &payload).is_ok());
        plan.points[9].dispatch_prefix.remove(LOCAL);
        receipt.records[9].dispatch_prefix.remove(LOCAL);
        receipt.plan_sha256 = digest(&serde_json::to_vec(&plan).unwrap());
        assert!(check(&plan, &receipt, &payload).is_err());
    }

    #[test]
    fn sg8_rejects_wrong_attention_kind_and_stale_dispatch_prefix() {
        let (mut plan, mut receipt, payload) = sg8_fixture();
        assert!(check(&plan, &receipt, &payload).is_ok());

        let base = LAYER_BOUNDARIES.len();
        for i in 8..LAYER_BOUNDARIES.len() {
            plan.points[base + i].dispatch_prefix.remove(GLOBAL);
            plan.points[base + i]
                .dispatch_prefix
                .insert(LOCAL.into(), 1);
            receipt.records[base + i].dispatch_prefix =
                plan.points[base + i].dispatch_prefix.clone();
        }
        receipt.plan_sha256 = digest(&serde_json::to_vec(&plan).unwrap());
        assert!(check(&plan, &receipt, &payload).is_err());

        let base = LAYER_BOUNDARIES.len();
        for i in 8..LAYER_BOUNDARIES.len() {
            plan.points[base + i].dispatch_prefix.remove(LOCAL);
            plan.points[base + i]
                .dispatch_prefix
                .insert(GLOBAL.into(), 1);
            receipt.records[base + i].dispatch_prefix =
                plan.points[base + i].dispatch_prefix.clone();
        }
        plan.points[base + 7]
            .dispatch_prefix
            .insert(GLOBAL.into(), 1);
        receipt.records[base + 7].dispatch_prefix = plan.points[base + 7].dispatch_prefix.clone();
        receipt.plan_sha256 = digest(&serde_json::to_vec(&plan).unwrap());
        assert!(check(&plan, &receipt, &payload).is_err());
    }

    #[test]
    fn paired_capture_reports_only_the_first_stored_boundary() {
        let off = fixture();
        let mut sg8 = sg8_fixture();
        let equal = paired(&off, &sg8).unwrap();
        assert_eq!(equal.differing_elements, 0);
        assert_eq!(equal.first_stored_difference, None);
        assert_eq!(equal.compared_points, 34);

        for record_index in [0, 3] {
            let Storage::Stored {
                offset,
                len,
                sha256,
            } = &mut sg8.1.records[record_index].storage
            else {
                panic!("test boundary must be stored")
            };
            let start = usize::try_from(*offset).unwrap();
            let end = start + usize::try_from(*len).unwrap();
            sg8.2[start] ^= 1;
            *sha256 = digest(&sg8.2[start..end]);
        }
        sg8.1.payload_sha256 = digest(&sg8.2);
        let result = paired(&off, &sg8).unwrap();
        assert_eq!(result.differing_elements, 2);
        assert_eq!(result.differing_boundaries.len(), 2);
        assert_eq!(
            result.first_stored_difference,
            Some(StoredDifference {
                layer: Some(4),
                boundary: Boundary::InputResidual,
                element_index: 0,
                off_bits: u32::from(u16::from_le_bytes([off.2[0], off.2[1]])),
                sg8_bits: u32::from(u16::from_le_bytes([sg8.2[0], sg8.2[1]])),
            })
        );
    }

    #[test]
    fn paired_capture_requires_same_source_and_logical_history() {
        let off = fixture();
        let mut sg8 = sg8_fixture();
        sg8.0.identity.model_sha256 = digest(b"different-model");
        sg8.1.plan_sha256 = digest(&serde_json::to_vec(&sg8.0).unwrap());
        assert!(check(&sg8.0, &sg8.1, &sg8.2).is_ok());
        assert!(paired(&off, &sg8).is_err());

        let mut sg8 = sg8_fixture();
        sg8.0.logical_kv[0].block_table.rotate_left(1);
        sg8.1.logical_kv = sg8.0.logical_kv.clone();
        sg8.1.plan_sha256 = digest(&serde_json::to_vec(&sg8.0).unwrap());
        assert!(check(&sg8.0, &sg8.1, &sg8.2).is_ok());
        assert!(paired(&off, &sg8).is_err());

        let mut sg8 = sg8_fixture();
        sg8.0.logical_kv[2].producer_layer = 4;
        sg8.1.logical_kv = sg8.0.logical_kv.clone();
        sg8.1.plan_sha256 = digest(&serde_json::to_vec(&sg8.0).unwrap());
        assert!(check(&sg8.0, &sg8.1, &sg8.2).is_ok());
        assert!(paired(&off, &sg8).is_err());
    }

    #[test]
    fn paired_capture_rejects_wrong_route_or_forged_payload() {
        let off = fixture();
        let mut sg8 = sg8_fixture();
        sg8.0.route = Route::Off;
        for (point, record) in sg8.0.points.iter_mut().zip(&mut sg8.1.records) {
            point.dispatch_prefix.clear();
            record.dispatch_prefix.clear();
        }
        sg8.1.plan_sha256 = digest(&serde_json::to_vec(&sg8.0).unwrap());
        assert!(check(&sg8.0, &sg8.1, &sg8.2).is_ok());
        assert!(paired(&off, &sg8).is_err());

        let mut sg8 = sg8_fixture();
        sg8.2[0] ^= 1;
        assert!(paired(&off, &sg8).is_err());
    }
}
