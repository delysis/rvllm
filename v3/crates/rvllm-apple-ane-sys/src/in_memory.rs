//! Owned, synchronous in-memory ANE kernels. Private API access stays here;
//! callers never receive pointers or access a surface while ANE is executing.
//!
//! API sequence cross-checked against maderix/ANE's MIT-licensed bridge.

use objc2::rc::{autoreleasepool, Retained};
use objc2::runtime::AnyObject;
use objc2::{class, msg_send};
use objc2_foundation::{NSData, NSError, NSString};
use std::ffi::c_void;
use std::path::PathBuf;
use std::ptr::NonNull;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;

#[path = "diagnostic_journal.rs"]
mod diagnostic_journal;
#[path = "source_staging.rs"]
mod source_staging;

const QOS: u32 = 21;

static COMPILE_BUDGET_USED: AtomicUsize = AtomicUsize::new(0);

/// Process-wide reserved compiler attempts, including failed attempts. Cache
/// hits consume no budget. This counter cannot be reset to bypass a limit.
pub fn compile_budget_used() -> usize {
    COMPILE_BUDGET_USED.load(Ordering::Relaxed)
}

fn reserve_compile(counter: &AtomicUsize, limit: usize) -> Result<(), String> {
    counter
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |used| {
            if used < limit {
                used.checked_add(1)
            } else {
                None
            }
        })
        .map(|_| ())
        .map_err(|_| format!("ANE process compile budget of {limit} exhausted"))
}

extern "C" {
    fn IOSurfaceCreate(properties: *const AnyObject) -> *mut c_void;
    fn IOSurfaceGetAllocSize(surface: *mut c_void) -> usize;
    fn IOSurfaceGetBaseAddress(surface: *mut c_void) -> *mut c_void;
    fn IOSurfaceLock(surface: *mut c_void, options: u32, seed: *mut u32) -> i32;
    fn IOSurfaceUnlock(surface: *mut c_void, options: u32, seed: *mut u32) -> i32;
    fn CFRelease(object: *const c_void);
    fn NSTemporaryDirectory() -> *mut NSString;
}

fn runtime_error(operation: &str, error: *mut NSError) -> String {
    // SAFETY: NSError is either nil or an autoreleased error from the immediately
    // preceding Objective-C call, within the same autorelease pool.
    let detail = unsafe { error.as_ref() }
        .map(|error| error.localizedDescription().to_string())
        .unwrap_or_else(|| "no error detail".into());
    format!("ANE {operation}: {detail}")
}

fn dictionary(entries: &[(&str, &AnyObject)]) -> Retained<AnyObject> {
    // SAFETY: Foundation copies keys and retains values on insertion.
    unsafe {
        let dict: Retained<AnyObject> = msg_send![class!(NSMutableDictionary), new];
        for (key, value) in entries {
            let key = NSString::from_str(key);
            let _: () = msg_send![&dict, setObject: *value, forKey: &*key];
        }
        dict
    }
}

struct Surface {
    object: Retained<AnyObject>,
    raw: NonNull<c_void>,
    bytes: usize,
}

impl Surface {
    fn new(bytes: usize) -> Result<Self, String> {
        if bytes == 0 || bytes > u32::MAX as usize {
            return Err("ANE surface size must be in 1..=u32::MAX".into());
        }
        let size = crate::create_ns_number_u64(bytes as u64);
        let one = crate::create_ns_number_u64(1);
        let zero = crate::create_ns_number_u64(0);
        let properties = dictionary(&[
            ("IOSurfaceWidth", &size),
            ("IOSurfaceHeight", &one),
            ("IOSurfaceBytesPerElement", &one),
            ("IOSurfaceBytesPerRow", &size),
            ("IOSurfaceAllocSize", &size),
            ("IOSurfacePixelFormat", &zero),
        ]);
        // SAFETY: NSDictionary is toll-free bridged to CFDictionary; values are
        // positive byte dimensions. The create-rule reference is owned by Self.
        let raw = NonNull::new(unsafe { IOSurfaceCreate(&*properties) })
            .ok_or("IOSurfaceCreate failed")?;
        // SAFETY: raw is a live IOSurface. The wrapper retains it independently.
        let object: Option<Retained<AnyObject>> = unsafe {
            msg_send![class!(_ANEIOSurfaceObject), objectWithIOSurface: raw.as_ptr().cast::<objc2_io_surface::IOSurfaceRef>()]
        };
        let Some(object) = object else {
            unsafe { CFRelease(raw.as_ptr()) };
            return Err("ANE IOSurface wrapper failed".into());
        };
        let surface = Self { object, raw, bytes };
        if unsafe { IOSurfaceGetAllocSize(raw.as_ptr()) } < bytes {
            return Err("IOSurface allocation is smaller than requested".into());
        }
        Ok(surface)
    }

    fn with_bytes<R>(
        &mut self,
        length: usize,
        read: bool,
        copy: impl FnOnce(&mut [u8]) -> R,
    ) -> Result<R, String> {
        if length != self.bytes {
            return Err(format!(
                "ANE I/O expected {} bytes, got {length}",
                self.bytes
            ));
        }
        let flags = u32::from(read);
        // SAFETY: &mut self excludes concurrent host access. Kernel evaluation
        // is synchronous and requires &mut the owning kernel. Allocation length
        // was checked at creation and is immutable.
        unsafe {
            let status = IOSurfaceLock(self.raw.as_ptr(), flags, std::ptr::null_mut());
            if status != 0 {
                return Err(format!("IOSurfaceLock: {status}"));
            }
            let base = IOSurfaceGetBaseAddress(self.raw.as_ptr()).cast::<u8>();
            let result =
                (!base.is_null()).then(|| copy(std::slice::from_raw_parts_mut(base, length)));
            let status = IOSurfaceUnlock(self.raw.as_ptr(), flags, std::ptr::null_mut());
            if base.is_null() || status != 0 {
                return Err(format!(
                    "IOSurface I/O failed: null={}, unlock={status}",
                    base.is_null()
                ));
            }
            result.ok_or_else(|| "IOSurface base address unavailable".into())
        }
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        // SAFETY: release exactly our create-rule reference. The Objective-C
        // wrapper and request retain their own references until their drops.
        unsafe { CFRelease(self.raw.as_ptr()) };
    }
}

struct ModelDirectory(PathBuf);

impl Drop for ModelDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

struct LoadedProgram {
    model: Retained<AnyObject>,
    options: Retained<AnyObject>,
    input_bytes: usize,
    output_bytes: usize,
    loaded: bool,
    _directory: ModelDirectory,
}

/// One compiled and loaded graph, shared by independently owned requests.
/// Cloning this handle neither compiles nor loads a model. The final owner
/// unloads it. Rc deliberately keeps all program/request access on one thread.
#[derive(Clone)]
pub struct AneInMemoryProgram {
    inner: Rc<LoadedProgram>,
}

/// Control explicit compiler calls. Cache hits are resolved by the installed
/// framework for this model descriptor and client identity. A failed cached
/// load is returned without recompiling or retrying evaluation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AneProgramCachePolicy {
    Compile,
    ReuseOrCompile,
    RequireExisting,
    /// Reuse cached graphs; atomically limit explicit compiler attempts across
    /// this process. Cache availability may change after offline provisioning.
    ReuseOrCompileUpTo(usize),
}

fn compiled_model_exists(model: &AnyObject) -> Result<bool, String> {
    let class = objc2::runtime::AnyClass::get(c"_ANEInMemoryModel")
        .ok_or("ANE in-memory model class unavailable")?;
    let method = class
        .instance_method(objc2::sel!(compiledModelExists))
        .ok_or("ANE compiledModelExists unavailable")?;
    if method.return_type().to_bytes() != b"B" || method.arguments_count() != 2 {
        return Err("ANE compiledModelExists ABI is unsupported".into());
    }
    // SAFETY: checked against this installed class, not a guessed private ABI.
    // `model` is a retained _ANEInMemoryModel inside its autorelease pool.
    Ok(unsafe { msg_send![model, compiledModelExists] })
}

/// ANE kernel with persistent FP16/FP32 byte I/O. The request and its surfaces
/// are exclusive to this kernel, while the compiled graph may be shared.
/// Evaluation requires exclusive access and completes before returning.
pub struct AneInMemoryKernel {
    // Drop our program ownership before the request/surfaces. If this is the
    // last owner, unload occurs while all resources are still alive. Other
    // owners may keep a loaded graph alive without any current requests.
    program: AneInMemoryProgram,
    request: Retained<AnyObject>,
    inputs: [Surface; 1],
    outputs: [Surface; 1],
}

impl AneInMemoryProgram {
    /// Compile MIL with one optional weight blob at `weights/weight.bin`.
    /// `input_bytes` and `output_bytes` describe the compiled tensor storage.
    /// Compiler/load errors are returned; there is no CPU or GPU fallback.
    pub fn compile(
        mil: &str,
        weights: &[u8],
        input_bytes: usize,
        output_bytes: usize,
    ) -> Result<Self, String> {
        Self::compile_tensors(mil, weights, &[input_bytes], &[output_bytes])
    }

    pub fn compile_with_cache_policy(
        mil: &str,
        weights: &[u8],
        input_bytes: usize,
        output_bytes: usize,
        policy: AneProgramCachePolicy,
    ) -> Result<Self, String> {
        Self::compile_tensors_with_policy(mil, weights, &[input_bytes], &[output_bytes], policy)
    }

    /// Compile a fixed set of tensors. Surface indices follow the MIL function
    /// arguments and return values. Every byte count must include row padding.
    /// Multiple tensors are currently quarantined after an AppleH16ANEInterface
    /// kernel panic in the attention experiment. See reports/ane-panic-20260914.md.
    pub fn compile_tensors(
        mil: &str,
        weights: &[u8],
        input_bytes: &[usize],
        output_bytes: &[usize],
    ) -> Result<Self, String> {
        Self::compile_tensors_with_policy(
            mil,
            weights,
            input_bytes,
            output_bytes,
            AneProgramCachePolicy::Compile,
        )
    }

    fn compile_tensors_with_policy(
        mil: &str,
        weights: &[u8],
        input_bytes: &[usize],
        output_bytes: &[usize],
        policy: AneProgramCachePolicy,
    ) -> Result<Self, String> {
        Self::compile_with_staging_lifetime(mil, weights, input_bytes, output_bytes, policy, false)
    }

    // The retention alternative is reachable only by this module's ignored
    // lifetime fixture. Production keeps its current source-cleanup policy.
    fn compile_with_staging_lifetime(
        mil: &str,
        weights: &[u8],
        input_bytes: &[usize],
        output_bytes: &[usize],
        policy: AneProgramCachePolicy,
        retain_staging: bool,
    ) -> Result<Self, String> {
        if input_bytes.is_empty() || output_bytes.is_empty() {
            return Err("ANE kernel requires inputs and outputs".into());
        }
        // This must precede framework loading, IOSurface creation and every
        // private API call. Do not turn a known machine reboot into a retry.
        if input_bytes.len() != 1 || output_bytes.len() != 1 {
            return Err("multi-tensor ANE execution quarantined after AppleH16ANEInterface kernel panic; see v3/reports/ane-panic-20260914.md".into());
        }
        if [input_bytes[0], output_bytes[0]]
            .into_iter()
            .any(|bytes| bytes == 0 || bytes > u32::MAX as usize)
        {
            return Err("ANE surface size must be in 1..=u32::MAX".into());
        }
        diagnostic_journal::record("compile_requested", None)?;
        static FRAMEWORK: OnceLock<Result<(), String>> = OnceLock::new();
        FRAMEWORK
            .get_or_init(|| {
                crate::load_frameworks()?;
                for name in [
                    c"_ANEInMemoryModelDescriptor",
                    c"_ANEInMemoryModel",
                    c"_ANERequest",
                    c"_ANEIOSurfaceObject",
                ] {
                    if objc2::runtime::AnyClass::get(name).is_none() {
                        return Err(format!("ANE class {name:?} unavailable"));
                    }
                }
                Ok(())
            })
            .clone()?;
        autoreleasepool(|_| {
            Self::compile_inner(
                mil,
                weights,
                input_bytes,
                output_bytes,
                policy,
                retain_staging,
            )
        })
    }

    fn compile_inner(
        mil: &str,
        weights: &[u8],
        input_bytes: &[usize],
        output_bytes: &[usize],
        policy: AneProgramCachePolicy,
        retain_staging: bool,
    ) -> Result<Self, String> {
        let mil_data = NSData::with_bytes(mil.as_bytes());
        let weight_data = NSData::with_bytes(weights);
        let offset = crate::create_ns_number_u64(0);
        let entry = dictionary(&[("offset", &offset), ("data", weight_data.as_ref())]);
        let weights_dict = if weights.is_empty() {
            dictionary(&[])
        } else {
            dictionary(&[("@model_path/weights/weight.bin", &entry)])
        };
        let options = dictionary(&[]);
        // SAFETY: selectors and ABI match the in-memory framework contract.
        // Inputs are retained throughout compile, error objects stay in pool.
        let model: Retained<AnyObject> = unsafe {
            let descriptor: Option<Retained<AnyObject>> = msg_send![
                class!(_ANEInMemoryModelDescriptor), modelWithMILText: &*mil_data,
                weights: &*weights_dict, optionsPlist: std::ptr::null::<AnyObject>()
            ];
            let descriptor = descriptor.ok_or("ANE model descriptor unavailable")?;
            let model: Option<Retained<AnyObject>> = msg_send![
                class!(_ANEInMemoryModel), inMemoryModelWithDescriptor: &*descriptor
            ];
            model.ok_or("ANE in-memory model unavailable")?
        };
        let identifier: Retained<NSString> = unsafe { msg_send![&model, hexStringIdentifier] };
        let identifier = identifier.to_string();
        if identifier.is_empty()
            || !identifier
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_')
        {
            return Err(format!(
                "ANE model identifier is not a safe path component: {identifier:?}"
            ));
        }
        let temporary = unsafe { NSTemporaryDirectory().as_ref() }
            .ok_or("NSTemporaryDirectory unavailable")?
            .to_string();
        let path = PathBuf::from(temporary).join(identifier);
        // Exclusive creation prevents this kernel from deleting another live
        // model's directory. Cleanup only starts after successful ownership.
        std::fs::create_dir(&path)
            .map_err(|e| format!("ANE model directory {}: {e}", path.display()))?;
        let directory = ModelDirectory(path);
        let model_id = directory.0.file_name().and_then(|s| s.to_str());
        diagnostic_journal::record("descriptor_created", model_id)?;
        let cached = if policy == AneProgramCachePolicy::Compile {
            false
        } else {
            let exists = compiled_model_exists(&model)?;
            diagnostic_journal::record(if exists { "cache_hit" } else { "cache_miss" }, model_id)?;
            if !exists && policy == AneProgramCachePolicy::RequireExisting {
                return Err("ANE model is absent from the current client's compiled cache".into());
            }
            exists
        };
        std::fs::create_dir(directory.0.join("weights")).map_err(|e| e.to_string())?;
        std::fs::write(directory.0.join("model.mil"), mil).map_err(|e| e.to_string())?;
        if !weights.is_empty() {
            std::fs::write(directory.0.join("weights/weight.bin"), weights)
                .map_err(|e| e.to_string())?;
        }
        let mut error: *mut NSError = std::ptr::null_mut();
        if cached {
            // On 24G84 these are exact source staging copies created by compile.
            // Hard links restore that layout without a second weight copy. The
            // framework cache query, not these files, establishes a cache hit.
            std::fs::hard_link(directory.0.join("model.mil"), directory.0.join("net.plist"))
                .map_err(|e| format!("ANE cached MIL staging: {e}"))?;
            if weights.is_empty() {
                std::fs::write(directory.0.join("data"), []).map_err(|e| e.to_string())?;
            } else {
                std::fs::hard_link(
                    directory.0.join("weights/weight.bin"),
                    directory.0.join("data"),
                )
                .map_err(|e| format!("ANE cached weight staging: {e}"))?;
            }
            diagnostic_journal::record("cached_source_staged", model_id)?;
        } else {
            let limit = match policy {
                AneProgramCachePolicy::ReuseOrCompileUpTo(limit) => limit,
                _ => usize::MAX,
            };
            reserve_compile(&COMPILE_BUDGET_USED, limit)?;
            diagnostic_journal::record("compile_begin", model_id)?;
            let compiled: bool = unsafe {
                msg_send![&model, compileWithQoS: QOS, options: &*options, error: &mut error]
            };
            if !compiled {
                let _ = diagnostic_journal::record("compile_failed", model_id);
                return Err(runtime_error("compile", error));
            }
            diagnostic_journal::record("compile_completed", model_id)?;
        }
        error = std::ptr::null_mut();
        diagnostic_journal::record("load_begin", model_id)?;
        let loaded: bool =
            unsafe { msg_send![&model, loadWithQoS: QOS, options: &*options, error: &mut error] };
        if !loaded {
            let _ = diagnostic_journal::record("load_failed", model_id);
            return Err(runtime_error("load", error));
        }
        let program = Self {
            inner: Rc::new(LoadedProgram {
                model,
                options,
                input_bytes: input_bytes[0],
                output_bytes: output_bytes[0],
                loaded: true,
                _directory: directory,
            }),
        };
        diagnostic_journal::record("load_completed", program.inner.model_id())?;
        if retain_staging {
            diagnostic_journal::record("source_staging_retained", program.inner.model_id())?;
            return Ok(program);
        }
        // The model is loaded and the descriptor retains its weight NSData.
        // Verify `data` against the complete caller source before unlinking
        // this redundant staging copy. Daemon-owned files are never touched.
        // Existing mappings/handles survive unlink; later cache restoration
        // recreates source staging before load. A mismatch fails closed.
        if source_staging::remove_verified_copy(&program.inner._directory.0.join("data"), weights)?
        {
            diagnostic_journal::record("source_data_removed", program.inner.model_id())?;
        }
        // Only this exclusively owned directory is touched. Cleanup failure
        // drops/unloads the program rather than returning a partial owner.
        if !weights.is_empty() {
            std::fs::remove_file(program.inner._directory.0.join("weights/weight.bin"))
                .map_err(|e| format!("ANE source-weight cleanup: {e}"))?;
            diagnostic_journal::record("source_weights_removed", program.inner.model_id())?;
        }
        Ok(program)
    }

    /// Allocate one independent request and persistent I/O surfaces. Weights
    /// carried in its input remain private to that request. The graph is not
    /// recompiled or reloaded, and can outlive this original program handle.
    pub fn create_request(&self) -> Result<AneInMemoryKernel, String> {
        autoreleasepool(|_| {
            let inputs = [Surface::new(self.inner.input_bytes)?];
            let outputs = [Surface::new(self.inner.output_bytes)?];
            diagnostic_journal::record("surfaces_created", self.inner.model_id())?;
            let input_objects = crate::create_ns_array(&[inputs[0].object.clone()]);
            let output_objects = crate::create_ns_array(&[outputs[0].object.clone()]);
            let zero = crate::create_ns_number_u64(0);
            let indices = crate::create_ns_array(&[zero.clone()]);
            // SAFETY: the loaded model, retained one-element surface arrays,
            // and scalar indices match the single-input/output request ABI.
            // The request retains its objects, and the kernel owns every
            // resource until synchronous evaluation and final unload finish.
            let request: Option<Retained<AnyObject>> = unsafe {
                msg_send![class!(_ANERequest), requestWithInputs: &*input_objects, inputIndices: &*indices,
                    outputs: &*output_objects, outputIndices: &*indices,
                    weightsBuffer: std::ptr::null::<AnyObject>(), perfStats: std::ptr::null::<AnyObject>(),
                    procedureIndex: &*zero]
            };
            let request = request.ok_or("ANE request creation failed")?;
            diagnostic_journal::record("request_created", self.inner.model_id())?;
            Ok(AneInMemoryKernel {
                program: self.clone(),
                request,
                inputs,
                outputs,
            })
        })
    }
}

impl AneInMemoryKernel {
    pub fn compile(
        mil: &str,
        weights: &[u8],
        input_bytes: usize,
        output_bytes: usize,
    ) -> Result<Self, String> {
        AneInMemoryProgram::compile(mil, weights, input_bytes, output_bytes)?.create_request()
    }

    pub fn compile_tensors(
        mil: &str,
        weights: &[u8],
        input_bytes: &[usize],
        output_bytes: &[usize],
    ) -> Result<Self, String> {
        AneInMemoryProgram::compile_tensors(mil, weights, input_bytes, output_bytes)?
            .create_request()
    }

    /// Stage bytes without allocation. The slice is not modified.
    pub fn write_input(&mut self, data: &[u8]) -> Result<(), String> {
        self.write_tensor(0, data)
    }

    pub fn read_output(&mut self, data: &mut [u8]) -> Result<(), String> {
        self.read_tensor(0, data)
    }

    pub fn write_tensor(&mut self, index: usize, data: &[u8]) -> Result<(), String> {
        self.inputs
            .get_mut(index)
            .ok_or("ANE input index out of bounds")?
            .with_bytes(data.len(), false, |surface| surface.copy_from_slice(data))
    }

    pub fn read_tensor(&mut self, index: usize, data: &mut [u8]) -> Result<(), String> {
        self.outputs
            .get_mut(index)
            .ok_or("ANE output index out of bounds")?
            .with_bytes(data.len(), true, |surface| data.copy_from_slice(surface))
    }

    /// Write equally sized items to a strided tensor without copying its other
    /// contents. Used to append one KV position to resident attention storage.
    pub fn write_tensor_strided(
        &mut self,
        index: usize,
        offset: usize,
        stride: usize,
        item_bytes: usize,
        data: &[u8],
    ) -> Result<(), String> {
        let surface = self
            .inputs
            .get_mut(index)
            .ok_or("ANE input index out of bounds")?;
        if item_bytes == 0 || data.is_empty() || data.len() % item_bytes != 0 || stride < item_bytes
        {
            return Err("ANE strided write has invalid item geometry".into());
        }
        let end = (data.len() / item_bytes - 1)
            .checked_mul(stride)
            .and_then(|n| n.checked_add(offset))
            .and_then(|n| n.checked_add(item_bytes))
            .ok_or("ANE strided write size overflow")?;
        if end > surface.bytes {
            return Err("ANE strided write exceeds tensor storage".into());
        }
        surface.with_bytes(surface.bytes, false, |storage| {
            for (i, item) in data.chunks_exact(item_bytes).enumerate() {
                let start = offset + i * stride;
                storage[start..start + item_bytes].copy_from_slice(item);
            }
        })
    }

    pub fn evaluate(&mut self) -> Result<(), String> {
        diagnostic_journal::record("evaluate_begin", self.model_id())?;
        autoreleasepool(|_| {
            let mut error: *mut NSError = std::ptr::null_mut();
            // SAFETY: all tensors/request/model remain owned until this
            // synchronous evaluation returns. &mut self serializes evaluation.
            let success: bool = unsafe {
                msg_send![&self.program.inner.model, evaluateWithQoS: QOS,
                options: &*self.program.inner.options, request: &*self.request, error: &mut error]
            };
            if success {
                diagnostic_journal::record("evaluate_completed", self.model_id())?;
                Ok(())
            } else {
                let _ = diagnostic_journal::record("evaluate_failed", self.model_id());
                Err(runtime_error("evaluate", error))
            }
        })
    }

    fn model_id(&self) -> Option<&str> {
        self.program.inner.model_id()
    }
}

impl LoadedProgram {
    fn model_id(&self) -> Option<&str> {
        self._directory.0.file_name().and_then(|s| s.to_str())
    }

    fn unload(&mut self) -> Result<(), String> {
        if !self.loaded {
            return Ok(());
        }
        // Never retry an unload that the driver rejected.
        self.loaded = false;
        let _ = diagnostic_journal::record("unload_begin", self.model_id());
        let result = autoreleasepool(|_| {
            let mut error: *mut NSError = std::ptr::null_mut();
            // SAFETY: the model loaded successfully and no evaluation can be
            // in flight. Unload before request, surfaces, and directory drop.
            let success: bool =
                unsafe { msg_send![&self.model, unloadWithQoS: QOS, error: &mut error] };
            if success {
                Ok(())
            } else {
                Err(runtime_error("unload", error))
            }
        });
        let stage = if result.is_ok() {
            "unload_completed"
        } else {
            "unload_failed"
        };
        let _ = diagnostic_journal::record(stage, self.model_id());
        let _ = diagnostic_journal::record("unload_returned", self.model_id());
        result
    }
}

impl Drop for LoadedProgram {
    fn drop(&mut self) {
        if let Err(error) = self.unload() {
            eprintln!("{error}; model_id={:?}", self.model_id());
        }
    }
}

#[cfg(test)]
#[path = "cache_lifetime_tests.rs"]
mod cache_lifetime_tests;

#[cfg(test)]
mod tests {
    #[test]
    fn process_compile_budget_is_atomic_and_never_wraps() {
        let counter = std::sync::atomic::AtomicUsize::new(0);
        assert!(super::reserve_compile(&counter, 0).is_err());
        let successes = std::thread::scope(|scope| {
            let workers: Vec<_> = (0..4)
                .map(|_| {
                    let counter = &counter;
                    scope.spawn(move || {
                        (0..32)
                            .filter(|_| super::reserve_compile(counter, 16).is_ok())
                            .count()
                    })
                })
                .collect();
            workers
                .into_iter()
                .map(|worker| worker.join().unwrap())
                .sum::<usize>()
        });
        assert_eq!(successes, 16);
        assert_eq!(counter.load(std::sync::atomic::Ordering::Relaxed), 16);
        assert!(super::reserve_compile(&counter, 16).is_err());
        counter.store(usize::MAX, std::sync::atomic::Ordering::Relaxed);
        assert!(super::reserve_compile(&counter, usize::MAX).is_err());
    }
    use super::*;

    fn cache_fixture(weight: u16) -> (String, Vec<u8>) {
        let mil = r#"program(1.3)
[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-version", "9.0"}})]
{
    func main<ios18>(tensor<fp16, [1, 32, 1, 1]> x) {
        string pad_type = const()[name = string("pad_type"), val = string("valid")];
        tensor<int32, [2]> strides = const()[name = string("strides"), val = tensor<int32, [2]>([1, 1])];
        tensor<int32, [4]> pad = const()[name = string("pad"), val = tensor<int32, [4]>([0, 0, 0, 0])];
        tensor<int32, [2]> dilations = const()[name = string("dilations"), val = tensor<int32, [2]>([1, 1])];
        int32 groups = const()[name = string("groups"), val = int32(1)];
        tensor<fp16, [32, 32, 1, 1]> W = const()[name = string("W"), val = tensor<fp16, [32, 32, 1, 1]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(64)))];
        tensor<fp16, [1, 32, 1, 1]> y = conv(dilations = dilations, groups = groups, pad = pad, pad_type = pad_type, strides = strides, weight = W, x = x)[name = string("rvllm_owned_cache_probe_20260914")];
    } -> (y);
}
"#;
        let mut blob = vec![0; 128 + 32 * 32 * 2];
        blob[0..4].copy_from_slice(&1_u32.to_le_bytes());
        blob[4..8].copy_from_slice(&2_u32.to_le_bytes());
        blob[64..68].copy_from_slice(&0xDEAD_BEEF_u32.to_le_bytes());
        blob[68..72].copy_from_slice(&1_u32.to_le_bytes());
        blob[72..80].copy_from_slice(&2048_u64.to_le_bytes());
        blob[80..88].copy_from_slice(&128_u64.to_le_bytes());
        for i in 0..32 {
            let offset = 128 + (i * 32 + i) * 2;
            blob[offset..offset + 2].copy_from_slice(&weight.to_le_bytes());
        }
        (mil.into(), blob)
    }

    fn checked_cached_kernel(policy: AneProgramCachePolicy) -> AneInMemoryKernel {
        let (mil, weights) = cache_fixture(0x3800); // diagonal FP16 0.5
        let program =
            AneInMemoryProgram::compile_with_cache_policy(&mil, &weights, 2048, 2048, policy)
                .unwrap();
        assert!(compiled_model_exists(&program.inner.model).unwrap());
        let mut kernel = program.create_request().unwrap();
        drop(program);
        for step in 0..3 {
            let mut input = vec![0_u8; 2048];
            for i in 0..32 {
                let value = if (i + step) % 2 == 0 {
                    0x3c00_u16
                } else {
                    0x4000_u16
                };
                input[i * 64..i * 64 + 2].copy_from_slice(&value.to_le_bytes());
            }
            kernel.write_input(&input).unwrap();
            kernel.evaluate().unwrap();
            let mut output = vec![0_u8; 2048];
            kernel.read_output(&mut output).unwrap();
            for i in 0..32 {
                let expected = if (i + step) % 2 == 0 {
                    0x3800_u16
                } else {
                    0x3c00_u16
                };
                assert_eq!(&output[i * 64..i * 64 + 2], &expected.to_le_bytes());
            }
        }
        kernel
    }

    #[test]
    #[ignore = "one owned 32-channel graph, two loads and six evaluations"]
    fn cache_compile_and_reload() {
        drop(checked_cached_kernel(AneProgramCachePolicy::ReuseOrCompile));
        drop(checked_cached_kernel(
            AneProgramCachePolicy::RequireExisting,
        ));
    }

    #[test]
    #[ignore = "load-only across processes; requires cache_compile_and_reload first"]
    fn cache_require_existing() {
        drop(checked_cached_kernel(
            AneProgramCachePolicy::RequireExisting,
        ));
    }

    #[test]
    #[ignore = "cache query only; changed weights must miss before any compile/load"]
    fn cache_different_weights_miss() {
        let (mil, weights) = cache_fixture(0x3400);
        let result = AneInMemoryProgram::compile_with_cache_policy(
            &mil,
            &weights,
            2048,
            2048,
            AneProgramCachePolicy::RequireExisting,
        );
        assert!(matches!(result, Err(message) if message.contains("absent")));
    }

    #[test]
    #[ignore = "purges only this test's unloaded disposable 32-channel model"]
    fn cache_purge_owned_fixture() {
        let kernel = checked_cached_kernel(AneProgramCachePolicy::RequireExisting);
        let model = kernel.program.inner.model.clone();
        drop(kernel); // final program owner unloads; every request/surface drops
        autoreleasepool(|_| {
            assert!(compiled_model_exists(&model).unwrap());
            let class = objc2::runtime::AnyClass::get(c"_ANEInMemoryModel").unwrap();
            let method = class
                .instance_method(objc2::sel!(purgeCompiledModel))
                .unwrap();
            assert_eq!(method.return_type().to_bytes(), b"v");
            assert_eq!(method.arguments_count(), 2);
            // SAFETY: locally verified void/no-argument ABI. This exact model
            // belongs only to these serial disposable tests, and is unloaded.
            let _: () = unsafe { msg_send![&model, purgeCompiledModel] };
            let exists = compiled_model_exists(&model).unwrap();
            println!("owned model cache exists immediately after purge: {exists}");
            assert!(
                !exists,
                "purge returned without removing this cache identity"
            );
        });
    }

    /// Read-only ABI/path inventory. No model, request, surface, compile, load,
    /// evaluation or purge is created/invoked by this diagnostic.
    #[test]
    #[ignore = "loads the private framework and reads cache-path class properties"]
    fn cache_interface_inventory() {
        use objc2::runtime::{AnyClass, Sel};
        use objc2::sel;

        crate::load_frameworks().unwrap();
        for (name, selectors) in [
            (
                c"_ANEInMemoryModel",
                &[
                    c"compiledModelExists",
                    c"purgeCompiledModel",
                    c"localModelPath",
                    c"modelURL",
                ][..],
            ),
            (
                c"_ANEClient",
                &[c"compiledModelExistsFor:", c"purgeCompiledModel:"][..],
            ),
        ] {
            let Some(class) = AnyClass::get(name) else {
                println!("class {name:?}: unavailable");
                continue;
            };
            for name in selectors {
                let selector = Sel::register(name);
                match class.instance_method(selector) {
                    Some(method) => println!(
                        "method {class:?} {name:?}: return={:?} arguments={:?}",
                        method.return_type(),
                        (0..method.arguments_count())
                            .map(|i| method
                                .argument_type(i)
                                .unwrap()
                                .to_string_lossy()
                                .into_owned())
                            .collect::<Vec<_>>()
                    ),
                    None => println!("method {class:?} {name:?}: unavailable"),
                }
            }
        }
        let Some(class) = AnyClass::get(c"_ANEStrings") else {
            println!("_ANEStrings unavailable");
            return;
        };
        macro_rules! read_path {
            ($name:ident) => {
                if let Some(method) = class.class_method(sel!($name)) {
                    println!(
                        "class method {}: return={:?} argc={}",
                        stringify!($name),
                        method.return_type(),
                        method.arguments_count()
                    );
                    if method.return_type().to_bytes() == b"@" && method.arguments_count() == 2 {
                        // SAFETY: the installed runtime reports an object-returning,
                        // zero-argument class method. Only the documented-source
                        // path/name getters below are invoked, never a mutator.
                        let value: Option<Retained<AnyObject>> = unsafe { msg_send![class, $name] };
                        if let Some(value) = value {
                            let description: Retained<NSString> =
                                unsafe { msg_send![&value, description] };
                            println!("path {}: {}", stringify!($name), description);
                        }
                    }
                }
            };
        }
        autoreleasepool(|_| {
            read_path!(buildSpecificModelDataVaultDirectory);
            read_path!(buildSpecificUserModelDataVaultDirectory);
            read_path!(modelDataVaultDirectory);
            read_path!(userModelDataVaultDirectory);
            read_path!(systemModelsCacheDirectory);
            read_path!(inMemoryModelCacheName);
            read_path!(modelBinaryName);
            read_path!(modelSourceStoreName);
        });
    }

    #[test]
    fn invalid_storage_contracts_fail_before_compilation() {
        // Deliberately invalid MIL: these errors must come from our contract
        // checks, before framework loading or any private compiler call.
        for (inputs, outputs, message) in [
            (vec![], vec![64], "requires inputs and outputs"),
            (vec![64], vec![], "requires inputs and outputs"),
            (vec![64, 64], vec![64], "multi-tensor"),
            (vec![64], vec![64, 64], "multi-tensor"),
            (vec![0], vec![64], "surface size"),
            (vec![64], vec![0], "surface size"),
            (vec![u32::MAX as usize + 1], vec![64], "surface size"),
            (vec![64], vec![u32::MAX as usize + 1], "surface size"),
        ] {
            for policy in [
                AneProgramCachePolicy::Compile,
                AneProgramCachePolicy::ReuseOrCompile,
                AneProgramCachePolicy::RequireExisting,
                AneProgramCachePolicy::ReuseOrCompileUpTo(16),
            ] {
                let result = AneInMemoryProgram::compile_tensors_with_policy(
                    "invalid MIL",
                    &[],
                    &inputs,
                    &outputs,
                    policy,
                );
                assert!(matches!(result, Err(error) if error.contains(message)));
            }
        }
    }
}
