//! Public-only Core ML compile, load, and prediction bindings.
//!
//! This crate is the shipping boundary for Core ML runtime calls on macOS and
//! iOS. It never links or dynamically loads a private Apple framework. A
//! compute-unit request is policy, not proof that the Neural Engine executed.

use std::fmt;
use std::path::{Path, PathBuf};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum CoreMlComputeUnits {
    All,
    CpuAndNeuralEngine,
}

impl CoreMlComputeUnits {
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::CpuAndNeuralEngine => "cpu_and_neural_engine",
        }
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    const fn raw(self) -> isize {
        match self {
            Self::All => 2,
            Self::CpuAndNeuralEngine => 3,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct CoreMlExecutionReport {
    pub requested_compute_units: CoreMlComputeUnits,
    /// Public compute-unit selection does not establish device placement.
    pub neural_engine_execution_verified: bool,
}

impl CoreMlExecutionReport {
    #[must_use]
    pub const fn requested_only(requested_compute_units: CoreMlComputeUnits) -> Self {
        Self {
            requested_compute_units,
            neural_engine_execution_verified: false,
        }
    }

    #[must_use]
    pub const fn honest_summary(self) -> &'static str {
        match self.requested_compute_units {
            CoreMlComputeUnits::All => {
                "Core ML all compute units requested; ANE execution is not verified"
            }
            CoreMlComputeUnits::CpuAndNeuralEngine => {
                "Core ML CPU+Neural Engine compute units requested; ANE execution is not verified"
            }
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CoreMlRuntimeError {
    UnsupportedPlatform,
    InvalidPath {
        path: PathBuf,
        reason: &'static str,
    },
    InvalidFeatureName {
        feature: &'static str,
    },
    InvalidShape {
        feature: String,
        reason: &'static str,
    },
    ElementCountMismatch {
        feature: String,
        expected: usize,
        actual: usize,
    },
    Runtime {
        operation: &'static str,
        detail: String,
    },
}

impl fmt::Display for CoreMlRuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedPlatform => {
                write!(formatter, "public Core ML runtime requires macOS or iOS")
            }
            Self::InvalidPath { path, reason } => {
                write!(
                    formatter,
                    "invalid Core ML path {}: {reason}",
                    path.display()
                )
            }
            Self::InvalidFeatureName { feature } => {
                write!(
                    formatter,
                    "Core ML {feature} feature name is empty or contains NUL"
                )
            }
            Self::InvalidShape { feature, reason } => {
                write!(formatter, "invalid Core ML shape for {feature:?}: {reason}")
            }
            Self::ElementCountMismatch {
                feature,
                expected,
                actual,
            } => write!(
                formatter,
                "Core ML feature {feature:?} expected {expected} elements, got {actual}"
            ),
            Self::Runtime { operation, detail } => {
                write!(formatter, "public Core ML {operation} failed: {detail}")
            }
        }
    }
}

impl std::error::Error for CoreMlRuntimeError {}

#[derive(Copy, Clone, Debug)]
pub struct CoreMlF32Input<'a> {
    pub name: &'a str,
    pub shape: &'a [i64],
    pub values: &'a [f32],
}

#[derive(Copy, Clone, Debug)]
pub struct CoreMlF32Output<'a> {
    pub name: &'a str,
    pub shape: &'a [i64],
}

pub struct PublicCoreMlModel {
    requested_compute_units: CoreMlComputeUnits,
    platform: platform::LoadedModel,
}

impl fmt::Debug for PublicCoreMlModel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PublicCoreMlModel")
            .field("requested_compute_units", &self.requested_compute_units)
            .finish_non_exhaustive()
    }
}

impl PublicCoreMlModel {
    pub fn load(
        compiled_model_path: &Path,
        requested_compute_units: CoreMlComputeUnits,
    ) -> Result<Self, CoreMlRuntimeError> {
        validate_path(compiled_model_path)?;
        let platform = platform::load_model(compiled_model_path, requested_compute_units)?;
        Ok(Self {
            requested_compute_units,
            platform,
        })
    }

    #[must_use]
    pub const fn execution_report(&self) -> CoreMlExecutionReport {
        CoreMlExecutionReport::requested_only(self.requested_compute_units)
    }

    pub fn predict_f32(
        &self,
        input: CoreMlF32Input<'_>,
        output: CoreMlF32Output<'_>,
    ) -> Result<Vec<f32>, CoreMlRuntimeError> {
        let output_count = validate_prediction_contract(input, output)?;
        platform::predict_f32(&self.platform, input, output, output_count)
    }
}

pub fn compile_model(source_model_path: &Path) -> Result<PathBuf, CoreMlRuntimeError> {
    validate_path(source_model_path)?;
    platform::compile_model(source_model_path)
}

fn validate_path(path: &Path) -> Result<(), CoreMlRuntimeError> {
    let Some(path_string) = path.to_str() else {
        return Err(CoreMlRuntimeError::InvalidPath {
            path: path.to_path_buf(),
            reason: "path is not UTF-8",
        });
    };
    if path_string.is_empty() || path_string.as_bytes().contains(&0) {
        return Err(CoreMlRuntimeError::InvalidPath {
            path: path.to_path_buf(),
            reason: "path is empty or contains NUL",
        });
    }
    Ok(())
}

fn validate_feature_name(feature: &'static str, name: &str) -> Result<(), CoreMlRuntimeError> {
    if name.is_empty() || name.as_bytes().contains(&0) {
        Err(CoreMlRuntimeError::InvalidFeatureName { feature })
    } else {
        Ok(())
    }
}

fn checked_element_count(feature: &str, shape: &[i64]) -> Result<usize, CoreMlRuntimeError> {
    if shape.is_empty() {
        return Err(CoreMlRuntimeError::InvalidShape {
            feature: feature.to_owned(),
            reason: "shape is empty",
        });
    }
    shape.iter().try_fold(1usize, |elements, dimension| {
        if *dimension <= 0 {
            return Err(CoreMlRuntimeError::InvalidShape {
                feature: feature.to_owned(),
                reason: "dimensions must be positive",
            });
        }
        let dimension =
            usize::try_from(*dimension).map_err(|_| CoreMlRuntimeError::InvalidShape {
                feature: feature.to_owned(),
                reason: "dimension exceeds the host address space",
            })?;
        elements
            .checked_mul(dimension)
            .ok_or_else(|| CoreMlRuntimeError::InvalidShape {
                feature: feature.to_owned(),
                reason: "element count overflows the host address space",
            })
    })
}

fn validate_prediction_contract(
    input: CoreMlF32Input<'_>,
    output: CoreMlF32Output<'_>,
) -> Result<usize, CoreMlRuntimeError> {
    validate_feature_name("input", input.name)?;
    validate_feature_name("output", output.name)?;
    let input_count = checked_element_count(input.name, input.shape)?;
    if input.values.len() != input_count {
        return Err(CoreMlRuntimeError::ElementCountMismatch {
            feature: input.name.to_owned(),
            expected: input_count,
            actual: input.values.len(),
        });
    }
    checked_element_count(output.name, output.shape)
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
mod platform {
    use super::{CoreMlComputeUnits, CoreMlF32Input, CoreMlF32Output, CoreMlRuntimeError};
    use objc2::rc::{autoreleasepool, Retained};
    use objc2::runtime::AnyObject;
    use objc2::{class, msg_send};
    use std::ffi::{CStr, CString};
    use std::path::{Path, PathBuf};
    use std::ptr;

    #[link(name = "CoreML", kind = "framework")]
    extern "C" {}

    #[link(name = "Foundation", kind = "framework")]
    extern "C" {}

    const ML_MULTI_ARRAY_DATA_TYPE_FLOAT32: isize = 65_568;

    pub struct LoadedModel {
        model: Retained<AnyObject>,
    }

    pub fn compile_model(source_model_path: &Path) -> Result<PathBuf, CoreMlRuntimeError> {
        autoreleasepool(|_| {
            let source_url = file_url(source_model_path)?;
            let mut error: *mut AnyObject = ptr::null_mut();
            let compiled_url: *mut AnyObject = unsafe {
                msg_send![
                    class!(MLModel),
                    compileModelAtURL: Retained::as_ptr(&source_url),
                    error: &mut error
                ]
            };
            if compiled_url.is_null() {
                return Err(runtime_error("compile", error));
            }
            path_from_url(compiled_url).ok_or_else(|| CoreMlRuntimeError::Runtime {
                operation: "compile",
                detail: "compiled model URL did not contain a UTF-8 path".to_owned(),
            })
        })
    }

    pub fn load_model(
        compiled_model_path: &Path,
        requested_compute_units: CoreMlComputeUnits,
    ) -> Result<LoadedModel, CoreMlRuntimeError> {
        autoreleasepool(|_| {
            let config_ptr: *mut AnyObject =
                unsafe { msg_send![class!(MLModelConfiguration), new] };
            let Some(config) = (unsafe { Retained::from_raw(config_ptr) }) else {
                return Err(CoreMlRuntimeError::Runtime {
                    operation: "load",
                    detail: "MLModelConfiguration allocation returned null".to_owned(),
                });
            };
            let _: () = unsafe {
                msg_send![
                    Retained::as_ptr(&config),
                    setComputeUnits: requested_compute_units.raw()
                ]
            };
            let compiled_url = file_url(compiled_model_path)?;
            let mut error: *mut AnyObject = ptr::null_mut();
            let model_ptr: *mut AnyObject = unsafe {
                msg_send![
                    class!(MLModel),
                    modelWithContentsOfURL: Retained::as_ptr(&compiled_url),
                    configuration: Retained::as_ptr(&config),
                    error: &mut error
                ]
            };
            let Some(model) = (unsafe { Retained::retain(model_ptr) }) else {
                return Err(runtime_error("load", error));
            };
            Ok(LoadedModel { model })
        })
    }

    pub fn predict_f32(
        loaded: &LoadedModel,
        input: CoreMlF32Input<'_>,
        output: CoreMlF32Output<'_>,
        output_count: usize,
    ) -> Result<Vec<f32>, CoreMlRuntimeError> {
        autoreleasepool(|_| {
            let input_array = multi_array(input.shape)?;
            for (index, value) in input.values.iter().enumerate() {
                let number_ptr: *mut AnyObject =
                    unsafe { msg_send![class!(NSNumber), numberWithFloat: *value] };
                if number_ptr.is_null() {
                    return Err(CoreMlRuntimeError::Runtime {
                        operation: "input preparation",
                        detail: format!("NSNumber allocation returned null at element {index}"),
                    });
                }
                let _: () = unsafe {
                    msg_send![
                        Retained::as_ptr(&input_array),
                        setObject: number_ptr,
                        atIndexedSubscript: index
                    ]
                };
            }

            let feature_value_ptr: *mut AnyObject = unsafe {
                msg_send![
                    class!(MLFeatureValue),
                    featureValueWithMultiArray: Retained::as_ptr(&input_array)
                ]
            };
            let Some(feature_value) = (unsafe { Retained::retain(feature_value_ptr) }) else {
                return Err(CoreMlRuntimeError::Runtime {
                    operation: "input preparation",
                    detail: "MLFeatureValue allocation returned null".to_owned(),
                });
            };
            let input_key = ns_string(input.name, "input")?;
            let dictionary_ptr: *mut AnyObject = unsafe {
                msg_send![
                    class!(NSDictionary),
                    dictionaryWithObject: Retained::as_ptr(&feature_value),
                    forKey: Retained::as_ptr(&input_key)
                ]
            };
            let Some(dictionary) = (unsafe { Retained::retain(dictionary_ptr) }) else {
                return Err(CoreMlRuntimeError::Runtime {
                    operation: "input preparation",
                    detail: "NSDictionary allocation returned null".to_owned(),
                });
            };
            let provider_alloc: *mut AnyObject =
                unsafe { msg_send![class!(MLDictionaryFeatureProvider), alloc] };
            let mut error: *mut AnyObject = ptr::null_mut();
            let provider_ptr: *mut AnyObject = unsafe {
                msg_send![
                    provider_alloc,
                    initWithDictionary: Retained::as_ptr(&dictionary),
                    error: &mut error
                ]
            };
            let Some(provider) = (unsafe { Retained::from_raw(provider_ptr) }) else {
                return Err(runtime_error("input provider creation", error));
            };

            let mut error: *mut AnyObject = ptr::null_mut();
            let output_provider_ptr: *mut AnyObject = unsafe {
                msg_send![
                    Retained::as_ptr(&loaded.model),
                    predictionFromFeatures: Retained::as_ptr(&provider),
                    error: &mut error
                ]
            };
            let Some(output_provider) = (unsafe { Retained::retain(output_provider_ptr) }) else {
                return Err(runtime_error("prediction", error));
            };
            let output_key = ns_string(output.name, "output")?;
            let output_feature_ptr: *mut AnyObject = unsafe {
                msg_send![
                    Retained::as_ptr(&output_provider),
                    featureValueForName: Retained::as_ptr(&output_key)
                ]
            };
            if output_feature_ptr.is_null() {
                return Err(CoreMlRuntimeError::Runtime {
                    operation: "prediction",
                    detail: format!("output feature {:?} was missing", output.name),
                });
            }
            let output_array_ptr: *mut AnyObject =
                unsafe { msg_send![output_feature_ptr, multiArrayValue] };
            if output_array_ptr.is_null() {
                return Err(CoreMlRuntimeError::Runtime {
                    operation: "prediction",
                    detail: format!("output feature {:?} was not an MLMultiArray", output.name),
                });
            }
            let actual_shape = multi_array_shape(output_array_ptr)?;
            if actual_shape != output.shape {
                return Err(CoreMlRuntimeError::Runtime {
                    operation: "prediction",
                    detail: format!(
                        "output feature {:?} shape mismatch: expected {:?}, got {:?}",
                        output.name, output.shape, actual_shape
                    ),
                });
            }
            let actual_count: usize = unsafe { msg_send![output_array_ptr, count] };
            if actual_count != output_count {
                return Err(CoreMlRuntimeError::ElementCountMismatch {
                    feature: output.name.to_owned(),
                    expected: output_count,
                    actual: actual_count,
                });
            }

            let mut values = Vec::with_capacity(output_count);
            for index in 0..output_count {
                let number_ptr: *mut AnyObject =
                    unsafe { msg_send![output_array_ptr, objectAtIndexedSubscript: index] };
                if number_ptr.is_null() {
                    return Err(CoreMlRuntimeError::Runtime {
                        operation: "prediction",
                        detail: format!("output element {index} was null"),
                    });
                }
                let value: f32 = unsafe { msg_send![number_ptr, floatValue] };
                values.push(value);
            }
            Ok(values)
        })
    }

    fn multi_array(shape: &[i64]) -> Result<Retained<AnyObject>, CoreMlRuntimeError> {
        let shape_numbers = shape
            .iter()
            .map(|dimension| ns_number_i64(*dimension))
            .collect::<Result<Vec<_>, _>>()?;
        let shape_array = ns_array(&shape_numbers)?;
        let allocation: *mut AnyObject = unsafe { msg_send![class!(MLMultiArray), alloc] };
        let mut error: *mut AnyObject = ptr::null_mut();
        let input_array_ptr: *mut AnyObject = unsafe {
            msg_send![
                allocation,
                initWithShape: Retained::as_ptr(&shape_array),
                dataType: ML_MULTI_ARRAY_DATA_TYPE_FLOAT32,
                error: &mut error
            ]
        };
        unsafe { Retained::from_raw(input_array_ptr) }
            .ok_or_else(|| runtime_error("multi-array allocation", error))
    }

    fn multi_array_shape(array: *mut AnyObject) -> Result<Vec<i64>, CoreMlRuntimeError> {
        let shape_ptr: *mut AnyObject = unsafe { msg_send![array, shape] };
        if shape_ptr.is_null() {
            return Err(CoreMlRuntimeError::Runtime {
                operation: "prediction",
                detail: "output MLMultiArray shape returned null".to_owned(),
            });
        }
        let rank: usize = unsafe { msg_send![shape_ptr, count] };
        let mut shape = Vec::with_capacity(rank);
        for index in 0..rank {
            let number_ptr: *mut AnyObject =
                unsafe { msg_send![shape_ptr, objectAtIndexedSubscript: index] };
            if number_ptr.is_null() {
                return Err(CoreMlRuntimeError::Runtime {
                    operation: "prediction",
                    detail: format!("output shape element {index} was null"),
                });
            }
            shape.push(unsafe { msg_send![number_ptr, longLongValue] });
        }
        Ok(shape)
    }

    fn ns_number_i64(value: i64) -> Result<Retained<AnyObject>, CoreMlRuntimeError> {
        let pointer: *mut AnyObject =
            unsafe { msg_send![class!(NSNumber), numberWithLongLong: value] };
        unsafe { Retained::retain(pointer) }.ok_or_else(|| CoreMlRuntimeError::Runtime {
            operation: "shape preparation",
            detail: "NSNumber allocation returned null".to_owned(),
        })
    }

    fn ns_array(
        objects: &[Retained<AnyObject>],
    ) -> Result<Retained<AnyObject>, CoreMlRuntimeError> {
        let pointers = objects.iter().map(Retained::as_ptr).collect::<Vec<_>>();
        let pointer: *mut AnyObject = unsafe {
            msg_send![
                class!(NSArray),
                arrayWithObjects: pointers.as_ptr(),
                count: pointers.len()
            ]
        };
        unsafe { Retained::retain(pointer) }.ok_or_else(|| CoreMlRuntimeError::Runtime {
            operation: "shape preparation",
            detail: "NSArray allocation returned null".to_owned(),
        })
    }

    fn ns_string(
        raw: &str,
        feature: &'static str,
    ) -> Result<Retained<AnyObject>, CoreMlRuntimeError> {
        let value =
            CString::new(raw).map_err(|_| CoreMlRuntimeError::InvalidFeatureName { feature })?;
        let pointer: *mut AnyObject =
            unsafe { msg_send![class!(NSString), stringWithUTF8String: value.as_ptr()] };
        unsafe { Retained::retain(pointer) }.ok_or_else(|| CoreMlRuntimeError::Runtime {
            operation: "string creation",
            detail: "NSString allocation returned null".to_owned(),
        })
    }

    fn file_url(path: &Path) -> Result<Retained<AnyObject>, CoreMlRuntimeError> {
        let path_string = path
            .to_str()
            .ok_or_else(|| CoreMlRuntimeError::InvalidPath {
                path: path.to_path_buf(),
                reason: "path is not UTF-8",
            })?;
        let path_ns = ns_string(path_string, "path")?;
        let pointer: *mut AnyObject = unsafe {
            msg_send![
                class!(NSURL),
                fileURLWithPath: Retained::as_ptr(&path_ns)
            ]
        };
        unsafe { Retained::retain(pointer) }.ok_or_else(|| CoreMlRuntimeError::Runtime {
            operation: "URL creation",
            detail: "NSURL allocation returned null".to_owned(),
        })
    }

    fn path_from_url(url: *mut AnyObject) -> Option<PathBuf> {
        let path_ns: *mut AnyObject = unsafe { msg_send![url, path] };
        if path_ns.is_null() {
            return None;
        }
        let utf8: *const std::ffi::c_char = unsafe { msg_send![path_ns, UTF8String] };
        if utf8.is_null() {
            return None;
        }
        Some(PathBuf::from(
            unsafe { CStr::from_ptr(utf8) }
                .to_string_lossy()
                .into_owned(),
        ))
    }

    fn runtime_error(operation: &'static str, error: *mut AnyObject) -> CoreMlRuntimeError {
        CoreMlRuntimeError::Runtime {
            operation,
            detail: objc_error_description(error)
                .unwrap_or_else(|| "Core ML returned no NSError detail".to_owned()),
        }
    }

    fn objc_error_description(error: *mut AnyObject) -> Option<String> {
        if error.is_null() {
            return None;
        }
        let description: *mut AnyObject = unsafe { msg_send![error, localizedDescription] };
        if description.is_null() {
            return None;
        }
        let utf8: *const std::ffi::c_char = unsafe { msg_send![description, UTF8String] };
        if utf8.is_null() {
            return None;
        }
        Some(
            unsafe { CStr::from_ptr(utf8) }
                .to_string_lossy()
                .into_owned(),
        )
    }
}

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
mod platform {
    use super::{CoreMlComputeUnits, CoreMlF32Input, CoreMlF32Output, CoreMlRuntimeError};
    use std::path::{Path, PathBuf};

    pub struct LoadedModel;

    pub fn compile_model(_source_model_path: &Path) -> Result<PathBuf, CoreMlRuntimeError> {
        Err(CoreMlRuntimeError::UnsupportedPlatform)
    }

    pub fn load_model(
        _compiled_model_path: &Path,
        _requested_compute_units: CoreMlComputeUnits,
    ) -> Result<LoadedModel, CoreMlRuntimeError> {
        Err(CoreMlRuntimeError::UnsupportedPlatform)
    }

    pub fn predict_f32(
        _loaded: &LoadedModel,
        _input: CoreMlF32Input<'_>,
        _output: CoreMlF32Output<'_>,
        _output_count: usize,
    ) -> Result<Vec<f32>, CoreMlRuntimeError> {
        Err(CoreMlRuntimeError::UnsupportedPlatform)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_unit_report_never_claims_neural_engine_execution() {
        for units in [
            CoreMlComputeUnits::All,
            CoreMlComputeUnits::CpuAndNeuralEngine,
        ] {
            let report = CoreMlExecutionReport::requested_only(units);
            assert!(!report.neural_engine_execution_verified);
            assert!(report.honest_summary().contains("not verified"));
        }
    }

    #[test]
    fn independent_input_and_output_shapes_are_validated() {
        assert_eq!(checked_element_count("x", &[3, 1, 2]), Ok(6));
        assert_eq!(checked_element_count("y", &[2, 1, 2]), Ok(4));
        assert!(matches!(
            checked_element_count("x", &[3, 0, 2]),
            Err(CoreMlRuntimeError::InvalidShape { .. })
        ));
    }

    #[test]
    fn caller_input_length_must_match_shape_before_platform_dispatch() {
        let result = validate_prediction_contract(
            CoreMlF32Input {
                name: "x",
                shape: &[3, 1, 2],
                values: &[1.0, 2.0],
            },
            CoreMlF32Output {
                name: "y",
                shape: &[2, 1, 2],
            },
        );
        assert!(matches!(
            result,
            Err(CoreMlRuntimeError::ElementCountMismatch {
                expected: 6,
                actual: 2,
                ..
            })
        ));
    }
}
