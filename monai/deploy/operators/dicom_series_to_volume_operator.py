# Copyright 2021-2026 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import logging
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from monai.deploy.utils.importutil import optional_import

apply_presentation_lut, _ = optional_import("pydicom.pixels", name="apply_presentation_lut")
apply_rescale, _ = optional_import("pydicom.pixels", name="apply_rescale")

from monai.deploy.core import ConditionType, Fragment, Operator, OperatorSpec
from monai.deploy.core.domain.dicom_series_selection import StudySelectedSeries
from monai.deploy.core.domain.image import Image

from . import decoder_nvimgcodec


class SliceGeometryError(ValueError):
    """Raised when a series' slice geometry cannot be represented as a single 3D volume."""


class DICOMSeriesToVolumeOperator(Operator):
    """This operator converts an instance of DICOMSeries into an Image object.

    The loaded Image Object can be used for further processing via other operators.
    The data array will be a 3D image NumPy array with index order of `DHW`.
    Channel is limited to 1 as of now, and `C` is absent in the NumPy array.

    Supports uniformly spaced acquisitions; refuses uniform-with-interior-gaps acquisitions
    and non-uniform acquisitions (irregular spacing) unconditionally. The slices of a series
    must also share Image Orientation Patient, Pixel Spacing, and in-plane dimensions;
    series that mix geometries are not supported.

    Named Input:
        study_selected_series_list: List of StudySelectedSeries.
    Named Output:
        image: Image object.
    """

    # Use constants instead of enums in monai to avoid dependency at this level.

    # Space keys for coordinate systems
    MONAI_UTIL_ENUMS_SPACEKEYS_RAS = "RAS"
    MONAI_UTIL_ENUMS_SPACEKEYS_LPS = "LPS"
    MONAI_TRANSFORMS_SPATIAL_METADATA_NAME = "space"
    METADATA_SPACE_RAS = {MONAI_TRANSFORMS_SPATIAL_METADATA_NAME: MONAI_UTIL_ENUMS_SPACEKEYS_RAS}
    METADATA_SPACE_LPS = {MONAI_TRANSFORMS_SPATIAL_METADATA_NAME: MONAI_UTIL_ENUMS_SPACEKEYS_LPS}
    ATTRIBUTE_NIFTI_AFFINE = "nifti_affine_transform"
    ATTRIBUTE_DICOM_AFFINE = "dicom_affine_transform"
    DEFAULT_DEPTH_PIXEL_SPACING = 1.0  # for CR image, similar to (Simple) ITK

    # Slice spacing categories
    SPACING_UNIFORM = "uniform"
    SPACING_UNIFORM_WITH_INTERIOR_GAPS = "uniform_with_interior_gaps"
    SPACING_IRREGULAR = "irregular"

    # Tolerances
    ORIENTATION_LOG_ATOL = 1e-6
    ORIENTATION_RAISE_ATOL = 1e-5
    COSINE_ORTHOGONALITY_ATOL = 1e-6
    COSINE_ORTHOGONALITY_RAISE_ATOL = 1e-4
    COSINE_NORM_ATOL = 1e-6
    COSINE_NORM_RAISE_ATOL = 1e-4
    PIXEL_SPACING_LOG_RTOL = 1e-6
    PIXEL_SPACING_RAISE_RTOL = 1e-4
    DUPLICATE_POSITION_ATOL_MM = 1e-4
    DETERMINANT_RTOL = 1e-9

    # Reporting
    DUPLICATE_REPORT_LIMIT = 5
    TRACE_HEAD_TAIL = 5

    def __init__(
        self,
        fragment: Fragment,
        *args,
        affine_lps_to_ras: bool = True,
        grid_residual_rtol: float = 0.1,
        slice_spacing_rtol: float = 0.01,
        **kwargs,
    ):
        """Create an instance for a containing application object.

        This operator converts instances of DICOMSeries into an Image object.
        The loaded Image Object can be used for further processing via other operators.
        The data array will be a 3D image NumPy array with index order of `DHW`.
        Channel is limited to 1 as of now, and `C` is absent in the NumPy array.

        This operator registers `nvimgcodec` based compressed pixel data decoder plugin with Pydicom
        at application startup to support and improve the performance of decoding DICOM files with compressed
        pixel data of in JPEG, JPEG 2000, and HTJ2K, irrespective of if python-gdcm, Python libjpg and openjpeg
        based decoder plugins are available at runtime.

        Registering the decoder plugin is all automatic and does not require any additional change in user's application
        except for adding a dependency on the `nvimgcodec-cu12` and `nvidia-nvjpeg2k-cu12` packages (suffix of
        cu12 means CUDA 12.0 though cu13 is also supported).

        Named Input:
            study_selected_series_list: List of StudySelectedSeries.
        Named Output:
            image: Image object.

        Args:
            fragment (Fragment): An instance of the Application class which is derived from Fragment.
            affine_lps_to_ras (bool): If true, the affine transform in the image metadata is RAS oriented,
                                      otherwise it is LPS oriented. Default is True.
            grid_residual_rtol (float): How far any slice may sit off the regular grid implied by
                                      the base spacing, as a fraction of that spacing. This is the
                                      cumulative drift check, as distinct from the pairwise gap
                                      comparison governed by slice_spacing_rtol. Default is 0.1, i.e.
                                      a tenth of a slice.
            slice_spacing_rtol (float): Relative tolerance for comparing adjacent slice spacings.
                                      Default is 0.01.
        """

        for name, value in (
            ("grid_residual_rtol", grid_residual_rtol),
            ("slice_spacing_rtol", slice_spacing_rtol),
        ):
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"{name} must be a positive finite number, got {value!r}") from None
            if not np.isfinite(numeric) or numeric <= 0.0:
                raise ValueError(f"{name} must be a positive finite number, got {value!r}")

        self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        self.input_name_series = "study_selected_series_list"
        self.output_name_image = "image"
        self.affine_lps_to_ras = affine_lps_to_ras
        self.grid_residual_rtol = float(grid_residual_rtol)
        self.slice_spacing_rtol = float(slice_spacing_rtol)

        if not decoder_nvimgcodec.register_as_decoder_plugin():
            self._logger.warning("The nvimgcodec decoder plugin did not register successfully.")

        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input(self.input_name_series)
        spec.output(self.output_name_image).condition(ConditionType.NONE)

    def compute(self, op_input, op_output, context):
        """Performs computation for this operator and handles I/O."""

        study_selected_series_list = op_input.receive(self.input_name_series)

        # TODO: need to get a solution to correctly annotate and consume multiple image outputs.
        # For now, only supports the one and only one selected series.
        image = self.convert_to_image(study_selected_series_list)
        op_output.emit(image, self.output_name_image)

    # ------------------------------------------------------------------ #
    # Logging Helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def _fmt_seq(cls, values: Sequence | np.ndarray, fmt: str = "{:.4f}") -> str:
        """Formats a sequence for logging, eliding the middle if it is long."""
        try:
            items = [fmt.format(float(v)) for v in values]
        except (TypeError, ValueError):
            items = [str(v) for v in values]
        n = cls.TRACE_HEAD_TAIL
        if len(items) <= 2 * n:
            return "[" + ", ".join(items) + "]"
        return "[" + ", ".join(items[:n]) + f", ... ({len(items) - 2 * n} more) ..., " + ", ".join(items[-n:]) + "]"

    @staticmethod
    def _first_absent_numbers(numbers: List[int], limit: int) -> List[int]:
        """Returns up to `limit` of the integers missing between min(numbers) and max(numbers), lowest first.

        Walks the gaps between the sorted values rather than enumerating the whole range: the
        range is bounded only by the values in the headers, not by the number of instances loaded.
        """
        absent: List[int] = []
        ordered = sorted(numbers)
        for low, high in zip(ordered, ordered[1:]):
            absent.extend(range(low + 1, min(high, low + 1 + limit - len(absent))))
            if len(absent) >= limit:
                break
        return absent

    @staticmethod
    def _fmt_absent_numbers(absent: List[int], absent_count: int) -> str:
        """Formats absent Instance Numbers, naming how many were not listed."""
        listed = ", ".join(str(n) for n in absent)
        if absent_count > len(absent):
            return f"{absent_count} in total, the first {len(absent)}: [{listed}, ...]"
        return f"[{listed}]"

    @staticmethod
    def _fmt_matrix(matrix: np.ndarray) -> str:
        """Formats a 4x4 affine for logging, one row per line."""
        return "\n".join("    [" + "  ".join(f"{v: 12.6f}" for v in row) + "]" for row in np.asarray(matrix))

    @staticmethod
    def _sop_uid(sop_instance) -> str:
        """Returns an instance's SOP Instance UID for logging, or a placeholder. Never raises."""
        try:
            native = sop_instance.get_native_sop_instance()
            return str(getattr(native, "SOPInstanceUID", "unknown"))
        except Exception:  # noqa: BLE001 - logging helper must not raise
            return "unknown"

    @staticmethod
    def _instance_number(sop_instance) -> str:
        """Returns an instance's Instance Number (0020,0013) for logging, or a placeholder. Never raises."""
        try:
            native = sop_instance.get_native_sop_instance()
            value = getattr(native, "InstanceNumber", None)
            return str(value) if value is not None else "unknown"
        except Exception:  # noqa: BLE001 - logging helper must not raise
            return "unknown"

    # ------------------------------------------------------------------ #
    # Core Logic
    # ------------------------------------------------------------------ #
    def convert_to_image(self, study_selected_series_list: List[StudySelectedSeries]) -> Union[Image, None]:
        """Extracts the pixel data from a DICOM Series and other attributes to create an Image object"""
        # For now, only supports the one and only one selected series.
        if not study_selected_series_list or len(study_selected_series_list) < 1:
            raise ValueError("Missing expected input 'study_selected_series_list'")

        for study_selected_series in study_selected_series_list:
            if not isinstance(study_selected_series, StudySelectedSeries):
                raise ValueError("Element in input is not expected type, 'StudySelectedSeries'.")
            selected_series = study_selected_series.selected_series[0]
            dicom_series = selected_series.series
            selection_name = selected_series.selection_name

            self.prepare_series(dicom_series)
            metadata = self.create_metadata(dicom_series)

            # Add to the metadata the DICOMStudy properties and selection metadata
            metadata.update(self._get_instance_properties(study_selected_series.study))
            selection_metadata = {"selection_name": selection_name}
            metadata.update(selection_metadata)

            self._add_affine_metadata(metadata, dicom_series)

            voxel_data = self.generate_voxel_data(dicom_series)

            # The volume is built from series._sop_instances, the same list the geometry was
            # derived from. If a fix ever lets those two drift apart, every array index
            # after the divergence maps to the wrong patient position, silently. Check it
            expected_planes = len(dicom_series.get_sop_instances())
            if voxel_data.ndim == 3 and voxel_data.shape[0] != expected_planes:
                raise SliceGeometryError(
                    f"Volume has {voxel_data.shape[0]} planes but the series holds {expected_planes} "
                    f"SOP instances after geometry analysis. The array and the affine would describe "
                    f"different stacks"
                )

            image = self.create_volumetric_image(voxel_data, metadata)

            # Now it is time to assign the converted image to the SelectedSeries obj
            selected_series.image = image

            # Break out since limited to one series/image for now
            break

        # TODO: This needs to be updated once allowed to output multiple Image objects
        return study_selected_series_list[0].selected_series[0].image

    def _add_affine_metadata(self, metadata: Dict, dicom_series) -> None:
        """Sets the affine and coordinate space in metadata based on the flag affine_lps_to_ras.

        If the flag is true, the NIFTI affine (RAS) is used, otherwise the DICOM affine (LPS) is used.
        A series without an affine (e.g. single-instance CR) gets the space key only.
        """
        if self.affine_lps_to_ras:
            attribute, space = self.ATTRIBUTE_NIFTI_AFFINE, self.METADATA_SPACE_RAS
        else:
            attribute, space = self.ATTRIBUTE_DICOM_AFFINE, self.METADATA_SPACE_LPS
        if hasattr(dicom_series, attribute):
            metadata["affine"] = getattr(dicom_series, attribute)
        metadata.update(space)

    def generate_voxel_data(self, series):
        """Applies rescale slope and rescale intercept to the pixels.

        Supports monochrome image only for now. Photometric Interpretation attribute,
        tag (0028,0004), is considered. Both MONOCHROME2 (IDENTITY) and MONOCHROME1 (INVERSE)
        result in an output image where the minimum sample value is intended to be displayed
        as black.

        Args:
            series: DICOM Series for which the pixel data needs to be extracted.

        Returns:
            A 3D numpy tensor representing the volumetric data.
        """

        def _get_rescaled_pixel_array(sop_instance):
            # Use pydicom utility to apply a modality lookup table or rescale operator to the pixel array.
            # The pydicom Dataset is required which can be obtained from the first slice's native SOP instance.
            # If Modality LUT is present the return array is of np.uint8 or np.uint16, and if Rescale
            # Intercept and Rescale Slope are present, np.float64.
            # If the pixel array is already in the correct type, the return array is the same as the input array.

            if not sop_instance:
                return np.array([])

            # pydicom's DS parser accepts "NaN" and "inf" in its default validation mode, and
            # apply_rescale multiplies through, so a non-finite Rescale Slope or Intercept would
            # yield an all-NaN or all-inf slice that every cast test below rejects, leaving a float64
            # plane that reaches inference without error. A zero-length value fails inside
            # apply_rescale as a generic error. Checked outside the try so the error is not rewrapped
            # as a generic rescale failure. Checked only when pydicom will apply the rescale: both
            # tags present and no usable Modality LUT Sequence (0028,3000). pydicom selects the LUT
            # with ds.get("ModalityLUTSequence"), so only a non-empty sequence takes precedence; an
            # absent or zero-item sequence falls through to the rescale tags. Absent rescale tags
            # are fine: apply_rescale then uses the Modality LUT, or leaves the stored values unchanged
            rescale_tags = (((0x0028, 0x1053), "Rescale Slope"), ((0x0028, 0x1052), "Rescale Intercept"))
            rescale_states = [(tag, keyword, *self._tag_state(sop_instance, tag, 1)) for tag, keyword in rescale_tags]
            modality_lut_state, _ = self._tag_state(sop_instance, (0x0028, 0x3000), 1)
            rescale_applies = all(state != "absent" for _, _, state, _ in rescale_states)
            if modality_lut_state in ("absent", "empty") and rescale_applies:
                for tag, keyword, state, raw in rescale_states:
                    if state in ("empty", "unparsable", "non-finite"):
                        raise ValueError(
                            f"{keyword} ({tag[0]:04X},{tag[1]:04X}) is {state}, stored as {raw}; the "
                            f"slice cannot be rescaled to modality values. SOP Instance UID "
                            f"{self._sop_uid(sop_instance)}, Instance Number {self._instance_number(sop_instance)}"
                        )

            native_sop = None
            try:
                native_sop = sop_instance.get_native_sop_instance()
                rescaled_pixel_data = apply_rescale(sop_instance.get_pixel_array(), native_sop)
                # In our use cases, pixel data will be interpreted as if MONOCHROME2, hence need to
                # apply the presentation lut.
                rescaled_pixel_data = apply_presentation_lut(rescaled_pixel_data, native_sop)
            except Exception as e:
                self._logger.error(f"Failed to apply rescale to DICOM volume: {e}")
                raise RuntimeError("Failed to apply rescale to DICOM volume.") from e

            # The following tests are expecting the array is already of the Numpy type.
            if rescaled_pixel_data.dtype == np.uint8 or rescaled_pixel_data.dtype == np.uint16:
                self._logger.debug("Rescaled pixel array is already of type uint8 or uint16.")
            # Check if casting to uint16 and back to float results in the same values.
            elif np.all(rescaled_pixel_data > 0) and np.array_equal(
                rescaled_pixel_data, rescaled_pixel_data.astype(np.uint16)
            ):
                self._logger.debug("Rescaled pixel array can be safely casted to uint16 with equivalence test.")
                rescaled_pixel_data = rescaled_pixel_data.astype(dtype=np.uint16)
            # Check if casting to int16 and back to float results in the same values.
            elif np.array_equal(rescaled_pixel_data, rescaled_pixel_data.astype(np.int16)):
                self._logger.debug("Rescaled pixel array can be safely casted to int16 with equivalence test.")
                rescaled_pixel_data = rescaled_pixel_data.astype(dtype=np.int16)
            # Check casting to float32 with equivalence test
            elif np.array_equal(rescaled_pixel_data, rescaled_pixel_data.astype(np.float32)):
                self._logger.debug("Rescaled pixel array can be safely casted to float32 with equivalence test.")
                rescaled_pixel_data = rescaled_pixel_data.astype(np.float32)
            else:
                self._logger.debug("Rescaled pixel data remains as of type float64.")

            return rescaled_pixel_data

        slices = series.get_sop_instances()
        # The sop_instance get_pixel_array() returns a 2D NumPy array with index order
        # of `HW`. The pixel array of all instances will be stacked along the first axis,
        # so the final 3D NumPy array will have index order of [DHW]. This is consistent
        # with the NumPy array returned from the ITK GetArrayViewFromImage on the image
        # loaded from the same DICOM series.
        # The below code loads all slice pixel data into a list of NumPy arrays in memory
        # before stacking them into a single 3D volume. This can be inefficient for series
        # with many slices.
        if not slices:
            self._logger.warning("No SOP instances to build voxel data from; returning an empty array")
            return np.array([])

        # Get shape and dtype from the first slice to pre-allocate numpy array.
        try:
            first_slice_pixel_array = _get_rescaled_pixel_array(slices[0])
            vol_shape = (len(slices),) + first_slice_pixel_array.shape
            dtype = first_slice_pixel_array.dtype
        except Exception as e:
            self._logger.error(f"Failed to get pixel array from the first slice: {e}")
            raise

        # Pre-allocate the volume data array.
        vol_data = np.empty(vol_shape, dtype=dtype)
        vol_data[0] = first_slice_pixel_array

        # Read subsequent slices directly into the pre-allocated array.
        for i, s in enumerate(slices[1:], 1):
            try:
                vol_data[i] = _get_rescaled_pixel_array(s)
            except Exception as e:
                self._logger.error(f"Failed to get pixel array from slice {i}: {e}")
                raise

        return vol_data

    def create_volumetric_image(self, vox_data, metadata):
        """Creates an instance of 3D image.

        Args:
            vox_data: A numpy array representing the volumetric data.
            metadata: DICOM attributes in a dictionary.

        Returns:
            An instance of Image object.
        """
        image = Image(vox_data, metadata)
        return image

    # ------------------------------------------------------------------ #
    # Geometry Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _get_tag_value(sop_instance, tag) -> Optional[list]:
        """Returns a DICOM tag's value as a list of floats, or None if absent, empty, unparsable, or non-finite."""
        try:
            data_element = sop_instance[tag]
        except (KeyError, TypeError):
            return None
        if data_element is None or data_element.value is None:
            return None
        value = data_element.value
        # Single-valued numeric elements (DS, IS) arrive as scalars, not sequences; iterating them raises
        # TypeError. Normalize to a list first
        if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
            value = [value]
        try:
            parsed = [float(v) for v in value]
        except (TypeError, ValueError):
            return None
        if not all(np.isfinite(v) for v in parsed):
            return None
        return parsed

    @staticmethod
    def _tag_state(sop_instance, tag, expected_length: int) -> Tuple[str, Optional[str]]:
        """Says why a numeric tag is unusable, for log and error messages only.

        Returns (state, raw), where raw is a short repr of the stored value, or None if absent.

          absent      the element is not in the dataset
          empty       the element is present with no value (zero-length)
          unparsable  the value is present but is not numeric
          non-finite  the value parses but contains NaN or infinity
          short       the value parses but has fewer than expected_length components
          ok          usable
        """
        try:
            data_element = sop_instance[tag]
        except (KeyError, TypeError):
            return "absent", None
        if data_element is None:
            return "absent", None
        value = data_element.value
        raw = repr(value)
        raw = raw if len(raw) <= 80 else raw[:77] + "..."
        if value is None or (hasattr(value, "__len__") and len(value) == 0):
            return "empty", raw
        if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
            value = [value]
        try:
            parsed = [float(v) for v in value]
        except (TypeError, ValueError):
            return "unparsable", raw
        if not all(np.isfinite(v) for v in parsed):
            return "non-finite", raw
        if len(parsed) < expected_length:
            return f"short: {len(parsed)} of {expected_length} values", raw
        return "ok", raw

    def _slice_normal(self, iop) -> np.ndarray:
        """Computes the slice normal from the raw Image Orientation Patient direction cosines.

        Image Orientation Patient (0020,0037) holds direction cosines, which are unit vectors / normal
        by definition but often arrive as rounded Decimal Strings (i.e. not exactly normal). Normalizing
        the direction cosines would mean potentially reporting geometry that is not present; instead the
        deviation is checked against COSINE_NORM_RAISE_ATOL in compute_affine_transform, and raw cosine
        values are used throughout.
        """
        row_cosine = np.asarray(iop[0:3], dtype=float)
        col_cosine = np.asarray(iop[3:6], dtype=float)
        normal: np.ndarray = np.cross(row_cosine, col_cosine)
        norm = float(np.linalg.norm(normal))
        orthogonality = float(np.dot(row_cosine, col_cosine))

        self._logger.info(f"Image Orientation Patient: {self._fmt_seq(iop[:6], '{:.10g}')}")

        self._logger.debug(
            f"Row Cosine (r): {self._fmt_seq(row_cosine, '{:.10g}')}, |r|={np.linalg.norm(row_cosine):.10f}"
        )
        self._logger.debug(
            f"Col Cosine (c): {self._fmt_seq(col_cosine, '{:.10g}')}, |c|={np.linalg.norm(col_cosine):.10f}"
        )
        self._logger.debug(f"Orthogonality: r ⋅ c = {orthogonality:.3e} (0 if perfectly orthogonal)")
        self._logger.info(
            f"Slice Normal (n): {self._fmt_seq(normal, '{:.10g}')}, "
            f"|n|={norm:.12f}. Direction the slice planes face"
        )

        if norm <= 0.0:
            raise SliceGeometryError(
                f"Image Orientation Patient does not define a slice plane; "
                f"row and column cosines are parallel or zero: {iop}"
            )

        if abs(orthogonality) > self.COSINE_ORTHOGONALITY_RAISE_ATOL:
            raise SliceGeometryError(
                f"Row and column direction cosines in Image Orientation Patient (0020,0037) are not "
                f"orthogonal: dot product {orthogonality:.6e}, which is "
                f"{np.degrees(np.arcsin(np.clip(abs(orthogonality), -1.0, 1.0))):.4f} °'s off "
                f"square, above the limit of {self.COSINE_ORTHOGONALITY_RAISE_ATOL}. The in-plane "
                f"axes of the affine are not orthogonal and the slice normal derived from "
                f"their cross product is not the true plane normal; no rescaling can fix this, "
                f"because which of the two cosines is wrong cannot be known. The header is most "
                f"likely corrupt, or the sender wrote Image Orientation Patient to very few "
                f"decimal places: {list(iop[:6])}"
            )

        if abs(orthogonality) > self.COSINE_ORTHOGONALITY_ATOL:
            self._logger.warning(
                f"Row and column direction cosines in Image Orientation Patient (0020,0037) are not "
                f"orthogonal (dot product {orthogonality:.6e}, log threshold {self.COSINE_ORTHOGONALITY_ATOL}, "
                f"raise threshold {self.COSINE_ORTHOGONALITY_RAISE_ATOL}); the slice normal is "
                f"approximate and the in-plane axes of the affine are not exactly orthogonal"
            )

        return normal

    def _validate_orientation_consistency(self, iop_list: List[list]) -> None:
        """Every SOP instance in the series must share the same image orientation.

        Slice ordering is by projection onto the slice normal. If instances carry different
        orientations then their projections are onto different axes and are not comparable.

        Deviations above ORIENTATION_LOG_ATOL are reported, because they mean the Image
        Orientation Patient strings are not byte-identical across the series. Only deviations
        above ORIENTATION_RAISE_ATOL are fatal, so that a series with rounding jitter is
        described rather than rejected.
        """
        reference = np.asarray(iop_list[0], dtype=float)
        deviations = []
        for index, iop in enumerate(iop_list[1:], start=1):
            deviation = float(np.max(np.abs(np.asarray(iop, dtype=float) - reference)))
            deviations.append(deviation)
            if deviation > self.ORIENTATION_RAISE_ATOL:
                raise SliceGeometryError(
                    f"Image Orientation Patient is not consistent across the series. "
                    f"Instance 0 has {list(reference)}, instance {index} has {list(iop)} "
                    f"(max component deviation {deviation:.6f}, limit {self.ORIENTATION_RAISE_ATOL}). "
                    f"The series may contain more than one acquisition or orientation"
                )

        if not deviations:
            return

        worst = max(deviations)
        self._logger.debug(
            f"Orientation consistent across {len(iop_list)} instances; "
            f"max component deviation from instance 0 = {worst:.3e} "
            f"(log threshold {self.ORIENTATION_LOG_ATOL}, raise threshold {self.ORIENTATION_RAISE_ATOL})"
        )
        if worst > self.ORIENTATION_LOG_ATOL:
            self._logger.warning(
                f"Image Orientation Patient varies across the series by up to {worst:.6e} per component. "
                f"Below the {self.ORIENTATION_RAISE_ATOL} rejection threshold, so the orientation of the "
                f"first instance is used for the normal and the affine, but the slice planes are not "
                f"exactly parallel"
            )

    @staticmethod
    def _pixel_spacing_unusable(ps: Optional[list]) -> bool:
        """True when Pixel Spacing cannot scale an affine column.

        Absent, short, non-finite and non-positive all qualify. A zero or NaN element has
        to be caught here rather than by the relative comparison in
        _validate_pixel_spacing_consistency: that comparison divides by the reference, so an
        all-zero series yields 0/0 = nan, and max(0.0, nan) returns 0.0 in Python -- the series
        passes silently and the affine comes out singular, with only invariants 5a/5b noticing
        and only as a warning.
        """
        if ps is None or len(ps) < 2:
            return True
        values = np.asarray(ps[:2], dtype=float)
        return bool(not np.all(np.isfinite(values)) or np.any(values <= 0.0))

    def _reject_unusable_pixel_spacing(self, ps_list: List[Optional[list]], sop_instances) -> None:
        """Raises if any instance's Pixel Spacing (0028,0030) cannot scale the affine."""
        missing = [i for i, ps in enumerate(ps_list) if self._pixel_spacing_unusable(ps)]
        if not missing:
            return

        def _reason(i: int) -> str:
            ps = ps_list[i]
            if ps is not None and len(ps) >= 2:
                return f"non-positive {ps[:2]}"
            state, raw = self._tag_state(sop_instances[i], (0x0028, 0x0030), 2)
            return state if raw is None else f"{state}, stored as {raw}"

        shown = missing[: self.DUPLICATE_REPORT_LIMIT]
        detail = "\n".join(
            f"  array index {i}: Pixel Spacing {_reason(i)}, "
            f"SOP Instance UID {self._sop_uid(sop_instances[i])}, "
            f"Instance Number {self._instance_number(sop_instances[i])}"
            for i in shown
        )
        if len(missing) > len(shown):
            detail += f"\n  ... and {len(missing) - len(shown)} more"
        raise SliceGeometryError(
            f"Pixel Spacing (0028,0030) is absent, malformed, or not a pair of positive "
            f"finite numbers on {len(missing)} of {len(ps_list)} instance(s) that carry "
            f"orientation and position. The affine scales its in-plane axes by this value, so "
            f"it cannot be built. The series may be malformed or combine more than one "
            f"acquisition\n{detail}"
        )

    def _validate_pixel_spacing_consistency(self, ps_list: List[Optional[list]], sop_instances) -> None:
        """Every SOP instance must declare the same in-plane Pixel Spacing (0028,0030).

        The affine takes the first slice's Pixel Spacing and applies it to every plane, so a
        slice whose in-plane spacing differs would be scaled wrongly.
        """

        self._reject_unusable_pixel_spacing(ps_list, sop_instances)

        # Every entry is present here: the check above raised on any absent Pixel Spacing, so this
        # keeps the array indices of ps_list and only narrows the type
        spacings: List[list] = [ps for ps in ps_list if ps is not None]
        reference = np.asarray(spacings[0][:2], dtype=float)
        worst = 0.0
        offenders: List[Tuple[int, np.ndarray, float]] = []
        for index in range(1, len(spacings)):
            ps = np.asarray(spacings[index][:2], dtype=float)
            rel = float(np.max(np.abs(ps - reference) / np.abs(reference)))
            worst = max(worst, rel)
            if rel > self.PIXEL_SPACING_RAISE_RTOL:
                offenders.append((index, ps, rel))

        if offenders:
            shown_offenders = offenders[: self.DUPLICATE_REPORT_LIMIT]
            detail = "\n".join(
                f"  array index {i}: Pixel Spacing {list(ps)} (relative deviation {rel:.3e}), "
                f"SOP Instance UID {self._sop_uid(sop_instances[i])}, "
                f"Instance Number {self._instance_number(sop_instances[i])}"
                for i, ps, rel in shown_offenders
            )
            if len(offenders) > len(shown_offenders):
                detail += f"\n  ... and {len(offenders) - len(shown_offenders)} more"
            raise SliceGeometryError(
                f"Pixel Spacing (0028,0030) is not consistent across the series: {len(offenders)} "
                f"instance(s) differ from the first slice's {list(reference)} by more than "
                f"{self.PIXEL_SPACING_RAISE_RTOL} (relative). The affine applies the first slice's "
                f"spacing to the whole volume, so the in-plane scale of the differing slice(s) would "
                f"be wrong. The series may combine more than one acquisition.\n{detail}"
            )

        if worst > self.PIXEL_SPACING_LOG_RTOL:
            self._logger.warning(
                f"Pixel Spacing varies across the series by up to {worst:.3e} (relative), below the "
                f"{self.PIXEL_SPACING_RAISE_RTOL} rejection threshold; the first slice's value is used "
                f"for the whole volume"
            )
        else:
            self._logger.debug(
                f"Pixel Spacing consistent across {len(ps_list)} instances; max relative "
                f"deviation {worst:.3e} (log {self.PIXEL_SPACING_LOG_RTOL}, raise {self.PIXEL_SPACING_RAISE_RTOL})"
            )

    def _validate_dimensions_consistency(self, dim_list: List[Optional[Tuple[int, int]]], sop_instances) -> None:
        """Every SOP instance must share Rows (0028,0010) and Columns (0028,0011).

        The volume is stacked into one array pre-allocated from the first slice's shape, so a
        slice of a different size would otherwise raise a bare NumPy broadcast error from inside
        generate_voxel_data, misattributed to a pixel-read failure.

        Instances that do not carry both tags are skipped rather than refused; the pixel decode in
        generate_voxel_data will fail on a genuinely unreadable instance.
        """
        present = [(i, d) for i, d in enumerate(dim_list) if d is not None]
        skipped = len(dim_list) - len(present)
        if skipped:
            self._logger.debug(f"{skipped} instance(s) lack Rows/Columns; excluded from the dimension check")
        if len(present) < 2:
            return

        _, reference = present[0]
        offenders = [(i, d) for i, d in present[1:] if d != reference]
        if offenders:
            shown = offenders[: self.DUPLICATE_REPORT_LIMIT]
            detail = "\n".join(
                f"  array index {i}: Rows x Columns {d[0]}x{d[1]}, "
                f"SOP Instance UID {self._sop_uid(sop_instances[i])}, "
                f"Instance Number {self._instance_number(sop_instances[i])}"
                for i, d in shown
            )
            if len(offenders) > len(shown):
                detail += f"\n  ... and {len(offenders) - len(shown)} more"
            raise SliceGeometryError(
                f"Rows (0028,0010) and Columns (0028,0011) are not consistent across the series: "
                f"{len(offenders)} instance(s) differ from the first slice's "
                f"{reference[0]}x{reference[1]}. All slices must share in-plane dimensions to stack "
                f"into one volume.\n{detail}"
            )

        self._logger.debug(f"Rows/Columns consistent across {len(present)} instances: {reference[0]}x{reference[1]}")

    def _analyze_instance_numbers(self, sop_instances) -> Optional[List[int]]:
        """Logs Instance Number (0020,0013) coverage and returns the absent numbers.

        For corroboration only; missing instances are determined from the slice positions alone,
        in _classify_spacing and _analyze_slice_geometry. Instance Number (0020,0013) is Type 2 tag
        and its definition does not promise contiguity, per-series scoping, or a particular start value,
        so a gap in the numbering is not sufficient evidence of a loss.

        Returns the Instance Numbers that fall inside the observed range but were not loaded, so
        that _report_spacing can name them when it accepts or refuses a deficient series. At most
        2 x TRACE_HEAD_TAIL are listed, lowest first; the warning logged here gives the total. Empty
        when the numbering is continuous; None when it cannot say, because the tag is absent,
        unusable or repeated, or missing from some instances. Note that instances missing from the
        start or end of a series are invisible here and to every other check in this operator.
        """
        numbers = []
        for sop_instance in sop_instances:
            try:
                value = sop_instance[0x0020, 0x0013].value
            except (KeyError, TypeError):
                continue
            if value is None:
                continue
            try:
                numbers.append(int(value))
            except (TypeError, ValueError):
                continue

        if not numbers:
            self._logger.debug("No usable Instance Number (0020,0013) values found")
            return None

        self._logger.debug(
            f"Instance Number: {len(numbers)} of {len(sop_instances)} instances carry the tag; "
            f"range {min(numbers)}..{max(numbers)}; in slice order {self._fmt_seq(numbers, '{:.0f}')}"
        )

        if len(numbers) < 2:
            return None

        # If there are untagged or repeated Instance Numbers, we cannot reliably determine missing slices
        untagged = len(sop_instances) - len(numbers)
        repeated = len(numbers) - len(set(numbers))
        if untagged or repeated:
            self._logger.debug(
                f"Instance Number (0020,0013) cannot corroborate missing slices: {untagged} "
                f"instance(s) without a usable value, {repeated} repeated value(s)"
            )
            return None

        expected = max(numbers) - min(numbers) + 1
        if expected == len(numbers):
            return []

        absent_count = expected - len(numbers)
        absent = self._first_absent_numbers(numbers, 2 * self.TRACE_HEAD_TAIL)
        self._logger.warning(
            f"Instance Number (0020,0013) ranges from {min(numbers)} to {max(numbers)}, "
            f"implying {expected} instances, but {len(sop_instances)} were loaded. "
            f"Instances may be missing from the series as delivered. "
            f"Absent instance numbers: {self._fmt_absent_numbers(absent, absent_count)}"
        )
        return absent

    def _log_reference_spacing_tags(self, sop_instances) -> list:
        """Logs SpacingBetweenSlices as declared and returns a sorted list of values.

        Purely informational. SpacingBetweenSlices is not used in the construction of the
        affine transformation, as this tag may be absent. SpacingBetweenSlices (spacing between
        adjacent slices) is a Type 3 tag.
        """
        spacings = set()
        for sop_instance in sop_instances:
            spacing = self._get_tag_value(sop_instance, (0x0018, 0x0088))
            if spacing:
                spacings.add(round(float(spacing[0]), 6))
        self._logger.debug(
            f"Declared SpacingBetweenSlices (0018,0088) across the series: "
            f"{sorted(spacings) if spacings else 'absent'}"
        )
        return sorted(spacings)

    def _compare_against_declared_spacing(self, spacing: float, declared_spacing: list) -> None:
        """Logs how the derived spacing compares with SpacingBetweenSlices."""

        # More than one distinct value means the series disagrees with itself about its own
        # through-plane geometry -- the same mixed-acquisition signal the orientation, Pixel
        # Spacing and dimension checks look for. Not fatal: SpacingBetweenSlices is not used
        # to build the affine, so the geometry stays sound, but the series is worth a look
        if len(declared_spacing) > 1:
            self._logger.warning(
                f"(0018,0088) SpacingBetweenSlices is not consistent across the series: {len(declared_spacing)} "
                f"distinct values {declared_spacing}. The series may combine more than one acquisition. "
                f"Tag is not used to build the affine, so this does not affect the geometry, but the "
                f"comparison is reported once per distinct value"
            )

        if declared_spacing:
            for declared in declared_spacing:
                delta = spacing - declared
                percent = f"{100.0 * delta / declared:+.4f}%" if declared else "n/a (declared zero)"
                self._logger.debug(
                    f"Derived spacing {spacing:.6f} mm vs declared "
                    f"SpacingBetweenSlices {declared:.6f} mm: difference {delta:+.6f} mm ({percent})"
                )
        else:
            self._logger.debug(
                f"SpacingBetweenSlices (0018,0088) is absent, so the derived spacing "
                f"{spacing:.6f} mm cannot be cross-checked against the sender's claim"
            )

    def _log_slice_geometry_summary(self, normal, row_cosine, step_vector, depth, measured: bool = True) -> None:
        """Logs how a series' slices sit in the patient, for every series that passes the geometry checks.
        With one slice, the step and shear are reported as not measured.

        Three independent quantities, each measured against a different reference:

          obliquity          the slice normal against the nearest patient axis. How far the slice
                             planes are tilted away from a true axial, coronal or sagittal
                             acquisition. Determined via Image Orientation Patient.
          in-plane rotation  the row cosine against that same axis set, after projecting the axis
                             into the slice plane. Does not affect through-plane geometry. Determined
                             via Image Orientation Patient.
          shear              the origin-to-origin step against the slice normal. Whether the stack
                             advances squarely through its planes or slides sideways as it goes.
                             Determined via Image Orientation Patient and Image Position Patient.
        """
        n = np.asarray(normal, dtype=float)
        n = n / np.linalg.norm(n)
        axis_index = int(np.argmax(np.abs(n)))
        axis = ("+" if n[axis_index] > 0 else "-") + "xyz"[axis_index]
        obliquity = float(np.degrees(np.arccos(np.clip(abs(float(n[axis_index])), -1.0, 1.0))))

        # Build an ORTHONORMAL in-plane reference frame before measuring. Projecting two patient
        # axes into the slice plane does not keep them perpendicular once the plane is tilted --
        # at 38.7 deg of obliquity they close to 80.4 deg -- so measuring against the nearer of
        # the two raw projections makes the answer depend on whether the row or the column cosine
        # was used. Orthonormalising first removes that: e1 is one patient axis projected in, and
        # e2 completes the frame, so the row and column cosines then agree to float noise
        other = [index for index in range(3) if index != axis_index]
        e1 = np.eye(3)[other[0]] - float(np.dot(np.eye(3)[other[0]], n)) * n
        e1 = e1 / np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        r = np.asarray(row_cosine, dtype=float)
        r = r / np.linalg.norm(r)
        # fold into [0, 45] deg: the grid has four-fold symmetry, so a 90 deg turn is the same grid
        raw_angle = float(np.degrees(np.arctan2(float(np.dot(r, e2)), float(np.dot(r, e1)))))
        in_plane_rotation = abs((raw_angle + 45.0) % 90.0 - 45.0)

        v = np.asarray(step_vector, dtype=float)
        step_length = float(np.linalg.norm(v))
        shear = float(np.linalg.norm(v - float(np.dot(n, v)) * n))
        # Derived from the shear vector itself, so the angle and the millimetres cannot disagree
        shear_angle = float(np.degrees(np.arctan2(shear, depth)))
        percent = (100.0 * shear / depth) if depth else float("nan")

        self._logger.info(f"Obliquity: slice planes are tilted away from the {axis} axis by {obliquity:.3f}°")
        self._logger.info(f"In-Plane Rotation: pixel grid is turned within the plane by {in_plane_rotation:.3f}°")
        if not measured:
            self._logger.info("Origin-to-Origin Step: not measured, only one slice carries geometry")
            self._logger.info("Shear: not measured, only one slice carries geometry")
            self._logger.info(f"Depth Pixel Spacing: {depth:.6f} mm, the single-slice default")
            return
        self._logger.info(
            f"Origin-to-Origin Step: {self._fmt_seq(v, '{:.6f}')}, Length: {step_length:.6f} mm. "
            f"Moving from slice k origin --> slice k+1 origin"
        )
        self._logger.info(
            f"Shear: origin-to-origin step leans off the normal by {shear_angle:.3f}° "
            f"({shear:.6f} mm per slice, {percent:.2f}% of the spacing)"
        )
        self._logger.info(
            f"Depth Pixel Spacing: perpendicular spacing between slice planes along the normal is {depth:.6f} mm"
        )

    def _classify_spacing(self, gaps: np.ndarray, spacing: float) -> Tuple[str, np.ndarray]:
        """Classifies adjacent slice spacings and returns (state, outlier_indices).

        Spacing classifications:

        * ``uniform`` -- every gap matches the estimated spacing within tolerance.
        * ``uniform_with_interior_gaps`` -- every interior gap is a whole multiple of the estimated spacing within
          tolerance. Consistent with instances missing from an otherwise regular grid.
        * ``irregular`` -- gaps are not whole multiples. No single spacing describes the series.
        """
        if spacing <= 0.0:
            raise SliceGeometryError(
                f"Base slice spacing estimated as {spacing:.9f} mm, which is not positive. "
                f"Slice positions may be corrupt or all coincident"
            )

        multiples = gaps / spacing
        uniform = np.isclose(multiples, 1.0, rtol=self.slice_spacing_rtol, atol=0.0)
        integral = np.isclose(multiples, np.round(multiples), rtol=self.slice_spacing_rtol, atol=0.0)

        self._logger.debug(
            f"Gap multiples of base spacing {spacing:.6f} mm: "
            f"{self._fmt_seq(multiples, '{:.4f}')}. "
            f"Within tolerance of 1.0: {int(uniform.sum())}/{uniform.size}; "
            f"whole multiples: {int(integral.sum())}/{integral.size}"
        )

        if uniform.all():
            return self.SPACING_UNIFORM, np.array([], dtype=int)

        outliers = np.nonzero(~uniform)[0]
        if integral.all():
            return self.SPACING_UNIFORM_WITH_INTERIOR_GAPS, outliers
        return self.SPACING_IRREGULAR, outliers

    @staticmethod
    def _acquisition_number(sop_instance) -> int:
        """Returns Acquisition Number (0020,0012) as an int, or -1 when it cannot be read.

        Deliberately total: a malformed optional tag must not break duplicate resolution. Note
        that ``Dataset.get`` returns the VALUE, so a default only applies when the tag is absent
        entirely -- a tag present but empty or None still has to be handled here.
        """
        try:
            native = sop_instance.get_native_sop_instance()
            value = native.get("AcquisitionNumber", None)
        except Exception:  # noqa: BLE001 - must never raise
            return -1
        if value is None:
            return -1
        if isinstance(value, str) and not value.strip():
            return -1
        try:
            return int(value)
        except (TypeError, ValueError):
            return -1

    def _reject_duplicate_positions(self, sop_instances, positions) -> None:
        """Raises if any two instances share a position along the slice normal.

        Always fatal - two distinct instances at one position are, by construction, two
        different volumes, and a 3D array has exactly one plane per position. Proceeding
        would mean arbitrarily discarding one of the instances.

        Nothing is pruned, so series._sop_instances, the positions and the IPPs cannot drift apart,
        and _analyze_instance_numbers sees the series as delivered rather than reporting removed
        duplicates as instances missing in transit.
        """
        gaps = np.diff(positions)
        positive = gaps[gaps > 0.0]
        scale = float(np.median(positive)) if positive.size else 1.0
        atol = max(self.slice_spacing_rtol * scale, self.DUPLICATE_POSITION_ATOL_MM)
        self._logger.debug(
            f"Duplicate-position tolerance {atol:.9f} mm = max({self.slice_spacing_rtol} x "
            f"median positive gap {scale:.6f} mm, floor {self.DUPLICATE_POSITION_ATOL_MM} mm)"
        )

        if not np.nonzero(gaps <= atol)[0].size:
            self._logger.debug("No duplicate slice positions detected")
            return

        # Group runs of coincident positions
        groups: List[List[int]] = []
        start = 0
        for i in range(gaps.size):
            if gaps[i] > atol:
                groups.append(list(range(start, i + 1)))
                start = i + 1
        groups.append(list(range(start, positions.size)))
        collided = [g for g in groups if len(g) > 1]

        surplus = sum(len(g) - 1 for g in collided)
        size_counts = sorted(collections.Counter(len(g) for g in collided).items())
        shape = ", ".join(f"{count} position(s) shared by {size} instances" for size, count in size_counts)

        # Summary
        shown = collided[: self.DUPLICATE_REPORT_LIMIT]
        detail = "\n".join(
            f"  Image Position Patient "
            f"{[self._get_tag_value(sop_instances[i], (0x0020, 0x0032)) for i in g]}: "
            f"{len(g)} instances at array indices {[int(i) for i in g]}, Acquisition Numbers "
            f"{[self._acquisition_number(sop_instances[i]) for i in g]}, "
            f"SOP Instance UIDs {[self._sop_uid(sop_instances[i]) for i in g]}"
            for g in shown
        )
        if len(collided) > len(shown):
            detail += f"\n  ... and {len(collided) - len(shown)} further position(s) not listed"

        raise SliceGeometryError(
            f"{len(collided)} position(s) along the slice normal carry more than one instance "
            f"({shape}), {surplus} instance(s) more than the {positions.size - surplus} distinct "
            f"positions can hold. A 3D volume has one plane per position, so this series cannot be "
            f"built without arbitrarily discarding instances, and nothing here can say which\n{detail}\n"
            f"If the Acquisition Numbers differ, the series may concatenate more than one acquisition. "
            f"If they are identical, the instances may be repeated exposures, or echoes or phases "
            f"of one acquisition"
        )

    def _analyze_slice_geometry(self, series) -> Dict:
        """Computes slice ordering, spacing and the inter-slice step vector for a series.

        Order of operations:
        1. Remove instances that lack Image Orientation Patient or Image Position Patient (absent
           or zero-length); refuse the series if either tag is present but unparsable, short,
           or carries a NaN or infinite value
        2. Refuse the series unless Image Orientation Patient, Pixel Spacing and Rows/Columns
           are consistent across every remaining instance
        3. Derive the slice normal from the first remaining instance (raw cosines used)
        4. Read and log SpacingBetweenSlices as declared
        5. Order the remaining instances by projection onto the slice normal
        6. Raise if two instances share a position
        7. Write `.distance` onto each instance and set series.depth_direction_cosine
        8. Report Instance Number coverage if present, and collect the numbers absent in transit
        9. Estimate the base spacing: median seed, then the mean of the gaps within
           slice_spacing_rtol of it
        10. Measure each slice's residual from the grid that spacing implies
        11. Classify spacing as uniform, uniform_with_interior_gaps or irregular
        12. Average the same inlier gaps as vectors to get the inter-slice step, preserving any
            in-plane shear, and project it onto the normal to get depth_pixel_spacing

        Acceptance is not decided here: this method measures and classifies, and _report_spacing
        applies the policy to what it returns.
        """
        sop_instances = series._sop_instances

        # Gather and validate per-instance geometry
        iop_list: List[list] = []
        ipp_list: List[list] = []
        ps_list: List[Optional[list]] = []
        dim_list: List[Optional[Tuple[int, int]]] = []
        keep: List[int] = []
        entry_count = len(sop_instances)
        removal_states: List[Tuple[str, str]] = []
        for index, sop_instance in enumerate(sop_instances):
            iop = self._get_tag_value(sop_instance, (0x0020, 0x0037))
            ipp = self._get_tag_value(sop_instance, (0x0020, 0x0032))
            if iop is None or len(iop) < 6 or ipp is None or len(ipp) < 3:
                iop_state, iop_raw = self._tag_state(sop_instance, (0x0020, 0x0037), 6)
                ipp_state, ipp_raw = self._tag_state(sop_instance, (0x0020, 0x0032), 3)
                # Removal is for instances that carry no plane, e.g. non-image objects filed in
                # the series, so only an absent or zero-length tag qualifies. A value that is
                # present but unparsable, short, or NaN/infinite is a corrupt header, not a missing
                # one: removing it would drop a real plane, invisibly if it sat at either end of the
                # stack, and leave a single-plane volume if it hit all but one instance
                if any(state not in ("absent", "empty", "ok") for state in (iop_state, ipp_state)):
                    raise SliceGeometryError(
                        f"Instance at index {index} carries a present but unusable geometry "
                        f"value: Image Orientation Patient (0020,0037) {iop_state}, Image Position "
                        f"Patient (0020,0032) {ipp_state}. IOP={iop_raw}, IPP={ipp_raw}, "
                        f"SOP Instance UID {self._sop_uid(sop_instance)}, "
                        f"Instance Number {self._instance_number(sop_instance)}"
                    )
                removal_states.append((iop_state, ipp_state))
                self._logger.warning(
                    f"Removing instance at index {index}: Image Orientation Patient (0020,0037) "
                    f"{iop_state}, Image Position Patient (0020,0032) {ipp_state}. "
                    f"IOP={iop_raw}, IPP={ipp_raw}, SOP Instance UID {self._sop_uid(sop_instance)}"
                )
                continue
            iop_list.append(iop)
            ipp_list.append(ipp)
            ps_list.append(self._get_tag_value(sop_instance, (0x0028, 0x0030)))
            rows = self._get_tag_value(sop_instance, (0x0028, 0x0010))
            cols = self._get_tag_value(sop_instance, (0x0028, 0x0011))
            dim_list.append((int(rows[0]), int(cols[0])) if rows and cols else None)
            keep.append(index)

        if len(keep) != len(sop_instances):
            self._logger.info(
                f"Removed {len(sop_instances) - len(keep)} instance(s) lacking usable geometry; {len(keep)} remain"
            )
            series._sop_instances = [sop_instances[i] for i in keep]
            sop_instances = series._sop_instances

        if len(sop_instances) < 1:
            if all(states == ("absent", "absent") for states in removal_states):
                raise SliceGeometryError(
                    f"None of the {entry_count} instances in this series carries Image Orientation "
                    f"Patient (0020,0037) or Image Position Patient (0020,0032) tags"
                )
            raise SliceGeometryError(
                f"None of the {entry_count} instances in this series carries usable Image "
                f"Orientation Patient and Image Position Patient tags, and at least one carries a tag "
                f"that is present but unusable; a volume cannot be constructed. The removal "
                f"warnings above give the state of each tag"
            )

        # Check orientation, in-plane spacing, and in-plane dimensions are consistent across the series
        self._validate_orientation_consistency(iop_list)
        self._validate_pixel_spacing_consistency(ps_list, sop_instances)
        self._validate_dimensions_consistency(dim_list, sop_instances)
        normal = self._slice_normal(iop_list[0])

        # Grab and SpacingBetweenSlices
        declared_spacing = self._log_reference_spacing_tags(sop_instances)

        # Order by projection onto the slice normal
        positions = np.array([float(np.dot(normal, np.asarray(p, dtype=float))) for p in ipp_list])
        self._logger.debug(f"Projections before sorting: {self._fmt_seq(positions)}")

        order = np.argsort(positions, kind="stable")

        series._sop_instances = [sop_instances[i] for i in order]
        sop_instances = series._sop_instances
        positions = positions[order]
        ipp_sorted = [np.asarray(ipp_list[i], dtype=float) for i in order]

        # Raise error if duplicate positions detected
        if len(sop_instances) > 1:
            self._reject_duplicate_positions(sop_instances, positions)

        self._logger.debug(f"Projections after sorting: {self._fmt_seq(positions)}")
        self._logger.debug(
            f"First slice IPP: {self._fmt_seq(ipp_sorted[0], '{:.6g}')}; "
            f"Last slice IPP: {self._fmt_seq(ipp_sorted[-1], '{:.6g}')}; "
            f"Span={positions[-1] - positions[0]:.4f} mm"
        )

        # Record ordering onto the surviving instances
        for sop_instance, position in zip(sop_instances, positions):
            sop_instance.distance = float(position)

        # `+ 0.0` maps -0.0 to 0.0. A cross product with a zero component yields negative
        # zero for half the orientations, which is arithmetically identical but serialises
        # as "-0.0" into the Image metadata and into anything written from it
        series.depth_direction_cosine = [float(v) + 0.0 for v in normal]
        absent_instance_numbers = self._analyze_instance_numbers(sop_instances)

        # single instance cases after filtering out unusable geometries
        if len(sop_instances) < 2:
            self._logger.warning(
                f"Only {len(sop_instances)} instance(s) in this series carry usable geometry, so no "
                f"slice spacing can be measured and depth_pixel_spacing falls back to the "
                f"{self.DEFAULT_DEPTH_PIXEL_SPACING} mm default. The series was delivered with more "
                f"than one instance, so the others were removed for absent or unusable Image Orientation "
                f"Patient or Image Position Patient values. The volume is a single plane and its "
                f"through-plane spacing is a convention, not a measurement"
            )
            # The 1.0 mm is a convention, not a measurement, so the step is built from the unit
            # normal to keep |col2| at exactly that convention, and depth is the projection onto
            # the raw normal as in the multi-slice path. Using the raw normal for both would make
            # depth |n|^2 mm and fail invariants 1 and 2 by twice the cosines' rounding
            unit_step = (normal / float(np.linalg.norm(normal))) * self.DEFAULT_DEPTH_PIXEL_SPACING
            return {
                "normal": normal,
                "row_cosine": iop_list[0][0:3],
                "positions": positions,
                "step_vector": unit_step,
                "depth_pixel_spacing": float(np.dot(normal, unit_step)),
                "regularity": self.SPACING_UNIFORM,
                "outlier_indices": np.array([], dtype=int),
                "gaps": np.array([]),
                "missing_slice_count": 0,
                "expected_slice_count": int(positions.size),
                "max_grid_residual": 0.0,
                "absent_instance_numbers": absent_instance_numbers,
                "ipp_sorted": ipp_sorted,
                "measured": False,
            }

        gaps = np.diff(positions)
        if gaps.size < 1:
            raise SliceGeometryError("Fewer than two distinct slice positions remain; cannot define a volume")

        # Slice spacing analysis
        unique_gaps, gap_counts = np.unique(np.round(gaps, 6), return_counts=True)
        self._logger.debug(
            f"{gaps.size} adjacent gaps; distinct values (6 dp): "
            + ", ".join(f"{g:.6f} mm x{c}" for g, c in zip(unique_gaps, gap_counts))
        )
        self._logger.debug(
            f"Gap statistics: min={float(gaps.min()):.6f} mm, max={float(gaps.max()):.6f} mm , "
            f"median={float(np.median(gaps)):.6f} mm, mean={float(gaps.mean()):.6f} mm, "
            f"std={float(gaps.std()):.3e} mm"
        )

        # DPS declaration - median first, then inlier mean of the gaps within tolerance of it
        # Median is immune to outlier gaps but inherits the float noise of a single observation;
        # averaging the inliers cancels that noise while staying robust
        spacing_seed = float(np.median(gaps))
        inlier_mask = np.isclose(gaps, spacing_seed, rtol=self.slice_spacing_rtol, atol=0.0)
        if not inlier_mask.any():
            self._logger.debug(
                f"No gap lies within {self.slice_spacing_rtol} of the median {spacing_seed:.9f} mm; "
                f"falling back to the median itself as the base spacing"
            )
            inlier_mask = np.isclose(gaps, spacing_seed, rtol=0.0, atol=np.inf)
        spacing = float(gaps[inlier_mask].mean())

        end_to_end_mean = float((positions[-1] - positions[0]) / (positions.size - 1))
        self._logger.debug(
            f"Spacing estimates: median seed={spacing_seed:.9f} mm, "
            f"inlier mean ({int(inlier_mask.sum())}/{inlier_mask.size} gaps)={spacing:.9f} mm, "
            f"end-to-end mean={end_to_end_mean:.9f} mm "
            f"(inflation {100.0 * (end_to_end_mean / spacing - 1.0):+.4f}% vs the value used), "
            f"first adjacent pair={float(gaps[0]):.9f} mm, "
            f"minimum gap={float(gaps.min()):.9f} mm"
        )

        # Compare derived spacings against DICOM tag values
        self._compare_against_declared_spacing(spacing, declared_spacing)

        # Compute residuals of each slice position relative to the ideal grid defined by base spacing
        # Similar to highdicom's "every position is a whole multiple of the base" check
        offsets = positions - positions[0]
        grid_indices = np.round(offsets / spacing)
        grid_residuals = np.abs(offsets - grid_indices * spacing)
        max_grid_residual = float(grid_residuals.max())
        if max_grid_residual <= 0.0:  # residuals are absolute values, so this means exactly on the grid
            self._logger.debug(
                f"Grid residual: all slices lie on the grid (residual 0); "
                f"tolerance {self.grid_residual_rtol} x {spacing:.6f} = "
                f"{self.grid_residual_rtol * spacing:.9f} mm"
            )
        else:
            worst_index = int(np.argmax(grid_residuals))
            self._logger.debug(
                f"Grid residual: worst slice is array index {worst_index}, "
                f"{max_grid_residual:.9f} mm ({max_grid_residual / spacing:.3e} of the base spacing) "
                f"from grid position {int(grid_indices[worst_index])}; "
                f"tolerance {self.grid_residual_rtol} x {spacing:.6f} = "
                f"{self.grid_residual_rtol * spacing:.9f} mm"
            )

        # Classify series spacing regularity and identify outliers
        regularity, outlier_indices = self._classify_spacing(gaps, spacing)

        # --- the step vector, preserving in-plane shear -------------------------------
        # The affine's third column is the vector from one slice origin to the next, not a scalar
        # along the normal. Under gantry tilt the two differ: the stack advances along the table
        # while the slice planes are tilted, so the step has an in-plane component. Averaging the
        # same inliers keeps that component while rejecting the gaps that span missing instances
        step_vectors = np.diff(np.stack(ipp_sorted, axis=0), axis=0)
        step_vector = step_vectors[inlier_mask].mean(axis=0)

        depth_pixel_spacing = float(np.dot(normal, step_vector))

        self._logger.debug(
            f"Step vector is the mean of the {int(inlier_mask.sum())} inlier gaps; scalar/vector "
            f"consistency (normal . mean step) - (mean of normal . step) = "
            f"{depth_pixel_spacing - spacing:+.3e} mm (must be float noise)"
        )

        if depth_pixel_spacing <= 0.0:
            raise SliceGeometryError(
                f"Computed a non-positive depth pixel spacing ({depth_pixel_spacing:.6f} mm) after sorting. "
                f"Slice positions may be corrupt"
            )

        # How far the affine's linear model lands from the last slice's actual position. Zero for
        # a uniform series; one base spacing per missing instance when gaps are present, since the
        # model advances by the base spacing per index and the real stack skipped positions
        modelled_last = ipp_sorted[0] + (len(ipp_sorted) - 1) * step_vector
        residual = float(np.linalg.norm(modelled_last - ipp_sorted[-1]))
        self._logger.debug(
            f"Step model check: last slice predicted at "
            f"{self._fmt_seq(modelled_last, '{:.4f}')}, actual {self._fmt_seq(ipp_sorted[-1], '{:.4f}')}, "
            f"residual {residual:.6f} mm"
        )

        # Missing instances
        missing_slice_count = 0
        if outlier_indices.size:
            multiples = np.round(gaps[outlier_indices] / spacing)
            missing_slice_count = int(max(0, int(np.sum(multiples - 1.0))))

        expected_slice_count = int(positions.size + missing_slice_count)

        return {
            "normal": normal,
            "row_cosine": iop_list[0][0:3],
            "positions": positions,
            "step_vector": step_vector,
            "depth_pixel_spacing": depth_pixel_spacing,
            "regularity": regularity,
            "outlier_indices": outlier_indices,
            "gaps": gaps,
            "missing_slice_count": missing_slice_count,
            "expected_slice_count": expected_slice_count,
            "max_grid_residual": max_grid_residual,
            "absent_instance_numbers": absent_instance_numbers,
            "ipp_sorted": ipp_sorted,
        }

    def _report_spacing(self, series, geometry: Dict) -> None:
        """Logs the spacing regularity and refuses any series that is not a complete grid.

        A volume is produced only when the present slices form a complete, uniformly spaced grid.
        The gate runs in a fixed order, most diagnostic first, so that a series failing several
        checks is reported by its root cause:

            1. irregular spacing - no single base spacing describes the series
            2. grid residual - the positions drift off the grid cumulatively
            3. uniform with interior gaps - interior instances are absent from an otherwise uniform grid
            4. uniform - accepted if every slice origin lies on the affine's lattice (_check_step_model)

        Logs the following diagnostics:
            slice_spacing_regularity: one of uniform | uniform_with_interior_gaps | irregular
            missing_slice_count: number of grid positions with no instance
            expected_slice_count: planes the volume would have if it were padded to the grid
                                  Equal to the plane count when the series is uniform
            slice_spacing_outliers: list of (index, gap_mm, position_before_mm, position_after_mm)
                                  where index is the ARRAY index of the slice BEFORE the gap, so the
                                  gap lies between indices index and index + 1 -- adjacent in the
                                  volume but not in the patient
            max_grid_residual_mm: worst distance of any slice from the regular grid
        """
        regularity = geometry["regularity"]
        spacing = geometry["depth_pixel_spacing"]
        gaps = geometry["gaps"]
        outliers = geometry["outlier_indices"]
        positions = geometry["positions"]
        missing = geometry["missing_slice_count"]
        residual = geometry["max_grid_residual"]
        absent_numbers = geometry["absent_instance_numbers"]

        slice_spacing_outliers = [
            (int(i), float(gaps[i]), float(positions[i]), float(positions[i + 1])) for i in outliers
        ]
        self._logger.debug(
            f"Slice geometry diagnostics: slice_spacing_regularity={regularity}, "
            f"missing_slice_count={missing}, expected_slice_count={geometry['expected_slice_count']}, "
            f"max_grid_residual_mm={float(residual):.9f}, "
            f"slice_spacing_outliers (each tuple = array index before the gap, gap mm, "
            f"position before mm, position after mm)={slice_spacing_outliers}"
        )

        detail = "\n".join(
            f"  gap {int(i)} -> {int(i) + 1}: {float(gaps[i]):.6f} mm "
            f"({float(gaps[i]) / spacing:.3f} x base) between positions "
            f"{float(positions[i]):.4f} and {float(positions[i + 1]):.4f} mm along the normal"
            for i in outliers
        )
        if absent_numbers is None:
            corroboration = (
                "\nInstance Number (0020,0013) is absent, repeated, or not carried by every "
                "instance, so it cannot be used to corroborate spacing regularity."
            )
        elif absent_numbers:
            corroboration = (
                f"\nInstance Numbers absent from the series as delivered include "
                f"{self._fmt_seq(absent_numbers, '{:.0f}')}. May be consistent with instances dropped "
                f"in transit rather than a non-uniform acquisition, though not proof of it: Instance "
                f"Numbers are not required to be contiguous."
            )
        else:
            corroboration = "\nInstance Numbers are continuous. Gaps may be due to a non-uniform acquisition."

        # Irregular spacing
        if regularity == self.SPACING_IRREGULAR:
            raise SliceGeometryError(
                f"Slice spacing is irregular: gaps are not whole multiples of the estimated base "
                f"spacing {spacing:.6f} mm.\n{detail}{corroboration}\n"
                f"The series cannot be represented as a regularly sampled volume"
            )

        # Grid residual check
        residual_limit = self.grid_residual_rtol * spacing
        if residual > residual_limit:
            raise SliceGeometryError(
                f"Slice positions drift off the regular grid: the worst slice sits "
                f"{residual:.6f} mm from its grid position, above the limit of "
                f"{residual_limit:.6f} mm ({self.grid_residual_rtol} x the {spacing:.6f} mm base "
                f"spacing). Each individual gap may still be within {self.slice_spacing_rtol} of "
                f"the base -- that is a pairwise test and this is the cumulative one.\n"
                f"Raise grid_residual_rtol to accept this series, or treat the positions as "
                f"unreliable"
            )

        # Regular spacing with interior gaps
        if regularity == self.SPACING_UNIFORM_WITH_INTERIOR_GAPS:
            raise SliceGeometryError(
                f"Slice spacing is uniform with interior gaps: base spacing {spacing:.6f} mm, "
                f"{outliers.size} gap(s) that are whole multiples of it, implying "
                f"{missing} missing instance(s) across {positions.size} present slices "
                f"({geometry['expected_slice_count']} grid positions).\n{detail}{corroboration}\n"
            )

        # Uniform spacing: the only classification left. The grid residual above tests positions
        # along the normal only; the affine also places each origin in-plane, so check that too
        self._check_step_model(geometry["ipp_sorted"], geometry["step_vector"], residual_limit)
        self._logger.debug(
            f"Slice spacing is uniform: {spacing:.6f} mm across {positions.size} slices "
            f"(worst grid residual {residual:.6f} mm)"
        )

    def _check_step_model(self, ipp_sorted: List[np.ndarray], step_vector: np.ndarray, limit: float) -> None:
        """Refuses a uniform series whose slice origins do not lie on the affine's lattice.

        The affine places slice k at IPP(first) + k x step_vector. The grid residual check sees
        only each origin's projection onto the slice normal, so an origin displaced within its
        plane passes it: IPPs (0,0,0), (10,0,4), (0,0,8) project to a uniform 4 mm, the averaged
        step is (0,0,4), and the middle plane would sit 10 mm from where the header puts it. A
        constant in-plane shear (gantry tilt) is part of step_vector and leaves no residual.
        Uses the same limit as the grid residual check, grid_residual_rtol x the base spacing.
        """
        origins = np.stack(ipp_sorted, axis=0)
        modelled = origins[0] + np.arange(origins.shape[0])[:, None] * step_vector
        residuals = np.linalg.norm(origins - modelled, axis=1)
        worst_index = int(np.argmax(residuals))
        worst = float(residuals[worst_index])
        self._logger.debug(
            f"Step model check: worst slice origin is array index {worst_index}, {worst:.6f} mm from "
            f"IPP(first) + index x step {self._fmt_seq(step_vector, '{:.6f}')}; limit {limit:.6f} mm"
        )
        if worst > limit:
            raise SliceGeometryError(
                f"Slice origins do not lie on a single straight lattice: array index {worst_index} sits "
                f"{worst:.6f} mm from IPP(first) + {worst_index} x the mean step "
                f"{self._fmt_seq(step_vector, '{:.6f}')} mm, above the limit of {limit:.6f} mm "
                f"({self.grid_residual_rtol} x the base spacing). The spacing along the normal is "
                f"uniform, so the displacement lies within the slice planes: the affine would place "
                f"that plane away from its stated position. The series may combine acquisitions with "
                f"different fields of view or table positions"
            )

    def prepare_series(self, series) -> None:
        """Orders the slices of a series and derives its depth pixel spacing and affine transforms.

        Each slice's position is projected onto the slice normal, giving a signed distance from
        the patient coordinate origin. Slices are ordered by that distance. The adjacent
        distances then yield both the depth pixel spacing and the inter-slice step vector used to
        build the affine.

        Args:
            series: An instance of DICOMSeries.
        """
        if not series._sop_instances:
            raise SliceGeometryError("The series contains no SOP instances")

        # Refuse multi-frame instances (e.g. Enhanced CT/MR)
        for index, sop_instance in enumerate(series._sop_instances):
            frames = self._get_tag_value(sop_instance, (0x0028, 0x0008))
            if frames is not None and len(frames) >= 1 and int(frames[0]) > 1:
                raise SliceGeometryError(
                    f"Instance at index {index} carries {int(frames[0])} frames (Number of Frames "
                    f"(0028,0008)). Multi-frame instances keep per-frame geometry in the functional "
                    f"group sequences, which this operator does not read; convert to single-frame "
                    f"instances before this operator. SOP Instance UID {self._sop_uid(sop_instance)}"
                )

        # Single-instance series (e.g. CR)
        if len(series._sop_instances) <= 1:
            self._prepare_single_instance_series(series)
            return

        # Detect non-images + duplicate positions, order by slice normal projection
        geometry = self._analyze_slice_geometry(series)

        # Classify spacing, enforce non-uniformity constraints
        self._report_spacing(series, geometry)
        series.depth_pixel_spacing = geometry["depth_pixel_spacing"]

        # Slice geometry summary
        self._log_slice_geometry_summary(
            geometry["normal"],
            geometry["row_cosine"],
            geometry["step_vector"],
            series.depth_pixel_spacing,
            measured=geometry.get("measured", True),
        )

        # Compute affine in DICOM and NIfTI coordinate systems
        self.compute_affine_transform(series._sop_instances[0], geometry["step_vector"], series)
        self.verify_affine_invariants(series)

    def _prepare_single_instance_series(self, series) -> None:
        """Handles a series of one instance, e.g. CR, where no spacing can be derived."""
        self._logger.info(
            f"Single-instance series; depth pixel spacing defaults to {self.DEFAULT_DEPTH_PIXEL_SPACING} mm"
        )
        series.depth_pixel_spacing = self.DEFAULT_DEPTH_PIXEL_SPACING
        self._logger.debug(
            f"Slice geometry diagnostics: slice_spacing_regularity={self.SPACING_UNIFORM}, "
            f"missing_slice_count=0, expected_slice_count=1, slice_spacing_outliers=[], "
            f"max_grid_residual_mm=0.0"
        )

        sop_instance = series._sop_instances[0]
        iop = self._get_tag_value(sop_instance, (0x0020, 0x0037))
        ipp = self._get_tag_value(sop_instance, (0x0020, 0x0032))

        if iop is None or len(iop) < 6:
            iop_state, iop_raw = self._tag_state(sop_instance, (0x0020, 0x0037), 6)
            if iop_state == "non-finite":
                raise SliceGeometryError(
                    f"Single-instance series has a non-finite (NaN or infinite) Image Orientation "
                    f"Patient (0020,0037): {iop_raw}. SOP Instance UID {self._sop_uid(sop_instance)}"
                )
            if iop_state == "absent":
                self._logger.info(
                    f"Single-instance series carries no Image Orientation Patient (0020,0037); "
                    f"no slice normal and no affine transform are computed. "
                    f"SOP Instance UID {self._sop_uid(sop_instance)}"
                )
            else:
                self._logger.warning(
                    f"Single-instance series has an unusable Image Orientation Patient (0020,0037) "
                    f"value ({iop_state}); no slice normal and no affine transform are computed. "
                    f"IOP={iop_raw}, SOP Instance UID {self._sop_uid(sop_instance)}"
                )
            return

        normal = self._slice_normal(iop)

        # `+ 0.0` maps -0.0 to 0.0. A cross product with a zero component yields negative
        # zero for half the orientations, which is arithmetically identical but serialises
        # as "-0.0" into the Image metadata and into anything written from it
        series.depth_direction_cosine = [float(v) + 0.0 for v in normal]

        if ipp is not None and len(ipp) >= 3:
            position = float(np.dot(normal, np.asarray(ipp[:3], dtype=float)))
            sop_instance.distance = position
            self._logger.debug(f"Single instance at {position:.6f} mm along the normal")
        else:
            ipp_state, ipp_raw = self._tag_state(sop_instance, (0x0020, 0x0032), 3)
            if ipp_state == "non-finite":
                raise SliceGeometryError(
                    f"Single-instance series has a non-finite (NaN or infinite) Image Position "
                    f"Patient (0020,0032): {ipp_raw}. SOP Instance UID {self._sop_uid(sop_instance)}"
                )
            if ipp_state == "absent":
                self._logger.info(
                    f"Single-instance series carries Image Orientation Patient but no Image Position "
                    f"Patient (0020,0032); no affine transform is computed. "
                    f"SOP Instance UID {self._sop_uid(sop_instance)}"
                )
            else:
                self._logger.warning(
                    f"Single-instance series has an unusable Image Position Patient (0020,0032) "
                    f"value ({ipp_state}); no affine transform is computed. IPP={ipp_raw}, "
                    f"SOP Instance UID {self._sop_uid(sop_instance)}"
                )
            return

        unit_step = (normal / float(np.linalg.norm(normal))) * self.DEFAULT_DEPTH_PIXEL_SPACING
        series.depth_pixel_spacing = float(np.dot(normal, unit_step))

        # Slice geometry summary
        self._log_slice_geometry_summary(normal, iop[0:3], unit_step, series.depth_pixel_spacing, measured=False)

        # Compute affine in DICOM and NIfTI coordinate systems
        self.compute_affine_transform(sop_instance, unit_step, series)
        self.verify_affine_invariants(series)

    def compute_affine_transform(self, s_1, step_vector, series):
        """Computes the affine transform for this series. It does it in both DICOM patient oriented
        coordinate system as well as the one preferred by NIFTI standard. Accordingly, the two attributes
        dicom_affine_transform and nifti_affine_transform are stored in the series instance.

        The Image Orientation Patient contains two triplets, [rx ry rz cx cy cz], which encode
        direction cosines of the row and column of an image slice. The Image Position Patient of the first
        slice in a volume, [x1 y1 z1], is the x, y, z coordinates of the centre of the first voxel transmitted.
        These two parameters define the location of the slice in PCS. To determine the location of a volume,
        the Image Position Patient of each slice is needed to determine the relative positions of the slices
        within the volume.

        To fully specify the slice geometry, one also needs the in-plane Pixel Spacing and the inter-slice step
        vector - these define the mapping from voxel indices to patient coordinates.

        The third column is the step vector itself rather than a scalar spacing times the slice
        normal, so that an acquisition whose slices advance at an angle to their own normal
        (i.e., gantry tilt) keeps its in-plane shear. `series.depth_pixel_spacing` is the projection of this
        same vector onto the slice normal as transmitted -- not rescaled to unit length, so that

            |det(m[:3, :3])| == row_axis_spacing * col_axis_spacing * depth_pixel_spacing

        stays exact for every orientation. Note that this invariant is symmetric in the two in-plane
        spacings and so cannot detect a transposed pairing. PixelSpacing[0] is the spacing between rows
        (it scales the column cosine) and PixelSpacing[1] between columns (it scales the row cosine);
        verify_affine_invariants checks 3 and 4 catch a swap.

        Args:
            s_1: The first slice in the ordered series.
            step_vector: The 3-vector from one slice origin to the next, in patient coordinates.
            series: An instance of DICOMSeries.
        """

        image_orientation_patient = self._get_tag_value(s_1, (0x0020, 0x0037))
        if image_orientation_patient is None or len(image_orientation_patient) < 6:
            raise SliceGeometryError(
                "Image Orientation Patient (0020,0037) is missing or malformed on the first slice; "
                "the affine transform cannot be computed"
            )

        row_cosine_raw = np.asarray(image_orientation_patient[0:3], dtype=float)
        col_cosine_raw = np.asarray(image_orientation_patient[3:6], dtype=float)
        row_norm = float(np.linalg.norm(row_cosine_raw))
        col_norm = float(np.linalg.norm(col_cosine_raw))
        if row_norm <= 0.0 or col_norm <= 0.0:
            raise SliceGeometryError(
                f"Image Orientation Patient (0020,0037) contains a zero-length direction cosine "
                f"on the first slice: row {list(row_cosine_raw)}, column {list(col_cosine_raw)}"
            )
        rx, ry, rz = (float(v) for v in row_cosine_raw)
        cx, cy, cz = (float(v) for v in col_cosine_raw)

        # Check direction cosine normals against the expected unit length
        worst_norm_error = max(abs(row_norm - 1.0), abs(col_norm - 1.0))
        if worst_norm_error > self.COSINE_NORM_RAISE_ATOL:
            raise SliceGeometryError(
                f"Image Orientation Patient direction cosines are not unit length as transmitted: "
                f"|r| = {row_norm:.12f}, |c| = {col_norm:.12f} (deviation {worst_norm_error:.3e}, "
                f"limit {self.COSINE_NORM_RAISE_ATOL})"
            )
        if worst_norm_error > self.COSINE_NORM_ATOL:
            self._logger.warning(
                f"Image Orientation Patient direction cosines are not unit length as transmitted: "
                f"|r| = {row_norm:.12f}, |c| = {col_norm:.12f} (deviation {worst_norm_error:.3e}, "
                f"reporting tolerance {self.COSINE_NORM_ATOL}). Within tolerance; proceeding"
            )

        # Extract and validate Pixel Spacing for the first slice
        pixel_spacing = self._get_tag_value(s_1, (0x0028, 0x0030))
        if pixel_spacing is None or self._pixel_spacing_unusable(pixel_spacing):
            # Reachable from the single-instance path, which does not run the series-wide Pixel
            # Spacing validation, so report which way the tag fails. A zero or negative element
            # parses cleanly ("ok") but would make the affine singular
            state, raw = self._tag_state(s_1, (0x0028, 0x0030), 2)
            state = "non-positive" if state == "ok" else state
            stored = "" if raw is None else f", stored as {raw}"
            raise SliceGeometryError(
                f"Pixel Spacing (0028,0030) on the first slice is {state}{stored}; "
                f"the affine transform cannot be computed"
            )

        # PixelSpacing is [between rows, between columns]. Row cosine advances along column
        # index (PixelSpacing[1]), column cosine advances along row index (PixelSpacing[0])
        # Both cosines are used as transmitted, so each column's norm is |cosine| * spacing
        # Also identifies non-square cases
        row_axis_spacing, col_axis_spacing = self._pair_pixel_spacing(pixel_spacing)

        # Extract and validate Image Position Patient for the first slice
        ip1 = self._get_tag_value(s_1, (0x0020, 0x0032))
        if ip1 is None or len(ip1) < 3:
            raise SliceGeometryError(
                "Image Position Patient (0020,0032) is missing or malformed on the first slice; "
                "the affine transform cannot be computed"
            )

        x1, y1, z1 = (float(v) for v in ip1[:3])
        sx, sy, sz = (float(v) for v in np.asarray(step_vector, dtype=float)[:3])

        self._logger.debug(
            f"Affine inputs: PixelSpacing=[{pixel_spacing[0]}, {pixel_spacing[1]}], "
            f"IOP={self._fmt_seq(image_orientation_patient[:6], '{:.10g}')}, "
            f"IPP(first)={self._fmt_seq(ip1[:3], '{:.6g}')}, "
            f"step={self._fmt_seq((sx, sy, sz), '{:.9f}')}"
        )

        # DICOM affine construction
        m1 = np.eye(4, dtype=float)

        m1[0, 0] = rx * row_axis_spacing
        m1[0, 1] = cx * col_axis_spacing
        m1[0, 2] = sx
        m1[0, 3] = x1

        m1[1, 0] = ry * row_axis_spacing
        m1[1, 1] = cy * col_axis_spacing
        m1[1, 2] = sy
        m1[1, 3] = y1

        m1[2, 0] = rz * row_axis_spacing
        m1[2, 1] = cz * col_axis_spacing
        m1[2, 2] = sz
        m1[2, 3] = z1

        # Every input is finite by now (_get_tag_value rejects NaN and infinity), but finite inputs
        # can still overflow: an IPP near 1e308 differences to inf in the step vector. The
        # invariant checks would see that only as a logged warning, so refuse it here, before
        # either affine is stored on the series
        if not np.all(np.isfinite(m1)):
            raise SliceGeometryError(
                f"The DICOM affine contains a non-finite value although every input tag was finite; "
                f"the geometry overflowed. Inputs: PixelSpacing={list(pixel_spacing[:2])}, "
                f"IPP(first)={list(ip1[:3])}, step={[sx, sy, sz]}\n{self._fmt_matrix(m1)}"
            )

        setattr(series, self.ATTRIBUTE_DICOM_AFFINE, m1)

        # NIfTI affine construction
        m2 = np.eye(4, dtype=float)

        m2[0, 0] = -rx * row_axis_spacing
        m2[0, 1] = -cx * col_axis_spacing
        m2[0, 2] = -sx
        m2[0, 3] = -x1

        m2[1, 0] = -ry * row_axis_spacing
        m2[1, 1] = -cy * col_axis_spacing
        m2[1, 2] = -sy
        m2[1, 3] = -y1

        m2[2, 0] = rz * row_axis_spacing
        m2[2, 1] = cz * col_axis_spacing
        m2[2, 2] = sz
        m2[2, 3] = z1

        setattr(series, self.ATTRIBUTE_NIFTI_AFFINE, m2)

        self._logger.debug(f"DICOM (LPS) affine:\n{self._fmt_matrix(m1)}")
        self._logger.debug(f"NIFTI (RAS) affine:\n{self._fmt_matrix(m2)}")

        # Compare cosines stored on series with the ones derived from the affine
        self._compare_with_stored_cosines(series, (rx, ry, rz), (cx, cy, cz))

        self._logger.debug(
            f"Affine column norms: |col0 (row cosine)|={float(np.linalg.norm(m1[:3, 0])):.9f} mm, "
            f"|col1 (col cosine)|={float(np.linalg.norm(m1[:3, 1])):.9f} mm, "
            f"|col2 (slice step)|={float(np.linalg.norm(m1[:3, 2])):.9f} mm"
        )

    def _compare_with_stored_cosines(self, series, row_cosine, col_cosine) -> None:
        """Reports when direction cosines differ from the ones stored on the series.

        DICOMDataLoaderOperator and DICOMSeriesToVolumeOperator both pull direction cosines
        from ImageOrientationPatient. The former sets series.row_direction_cosine and
        col_direction_cosine using the first instance os.listdir returns for the series, while the latter
        constructs the affine using the first SOP instance after sorting by projection onto the slice normal.

        Logs above ORIENTATION_LOG_ATOL but never raises: _validate_orientation_consistency has
        already bounded the whole series to ORIENTATION_RAISE_ATOL, so anything seen here is
        within a tolerance the series was already accepted under.
        """
        for name, attribute, used in (
            ("row", "row_direction_cosine", np.asarray(row_cosine, dtype=float)),
            ("column", "col_direction_cosine", np.asarray(col_cosine, dtype=float)),
        ):
            stored = getattr(series, attribute, None)
            if stored is None:
                continue
            try:
                stored_arr = np.asarray([float(v) for v in stored], dtype=float)
            except (TypeError, ValueError):
                self._logger.debug(f"series.{attribute} is not a usable triplet; comparison skipped")
                continue
            if stored_arr.size != 3:
                continue
            deviation = float(np.max(np.abs(stored_arr - used)))
            if deviation > self.ORIENTATION_LOG_ATOL:
                self._logger.warning(
                    f"The {name} direction cosine stored on the series differs from the one this "
                    f"affine was built from by {deviation:.3e} per component. The series attribute "
                    f"comes from the first instance the loader encountered; the affine comes from "
                    f"the first instance in sorted order. Both are within the "
                    f"{self.ORIENTATION_RAISE_ATOL} orientation-consistency limit, so the series is "
                    f"accepted, but any consumer reading series.{attribute} alongside this affine "
                    f"will combine a rotation and an origin taken from different slices"
                )
            else:
                self._logger.debug(
                    f"series.{attribute} matches the affine's {name} direction to "
                    f"{deviation:.3e} (log threshold {self.ORIENTATION_LOG_ATOL})"
                )

    def _pair_pixel_spacing(self, pixel_spacing: Sequence, log: bool = True) -> Tuple[float, float]:
        """Decides which Pixel Spacing element scales which in-plane affine column.

        Returns (row_axis_spacing, col_axis_spacing).

        PS3.3 C.7.6.2 defines Pixel Spacing (0028,0030) as "adjacent row spacing \\ adjacent
        column spacing", so PixelSpacing[0] is the spacing between adjacent rows and
        PixelSpacing[1] the spacing between adjacent columns (further explained in PS3.3 10.7.1.3).
        The pairing is crossed relative to those names: the row direction cosine runs along a row,
        i.e. the direction the column index advances, so a one-column step moves by the between-columns
        spacing PixelSpacing[1]; the column cosine runs along a column and pairs with the between-rows spacing
        PixelSpacing[0]. This is equation C.7.6.2.1-1 of PS3.3 C.7.6.2.1.1, where the row cosine X
        is scaled by the column pixel resolution dI = PixelSpacing[1] and the column cosine Y by
        the row pixel resolution dJ = PixelSpacing[0].

        Args:
            pixel_spacing: The Pixel Spacing (0028,0030) value, as [between rows, between columns].
            log (bool): Emit the pairing trace line and the non-square-pixel report. The method is
                        pure and is called twice per series -- once to build the affine and once by
                        verify_affine_invariants to re-derive it independently -- so the verifier
                        passes False and the decision is reported exactly once per series.
        """
        spacing_between_rows = float(pixel_spacing[0])
        spacing_between_cols = float(pixel_spacing[1])

        # Row cosine pairs with between-columns spacing, column cosine with between-rows spacing
        row_axis_spacing = spacing_between_cols
        col_axis_spacing = spacing_between_rows

        if not log:
            return row_axis_spacing, col_axis_spacing

        self._logger.debug(
            f"Pixel Spacing (0028,0030) = [{spacing_between_rows}, {spacing_between_cols}]: "
            f"between rows {spacing_between_rows} mm, between columns {spacing_between_cols} mm. "
            f"Row cosine scaled by {row_axis_spacing} (between columns, PixelSpacing[1]), "
            f"column cosine scaled by {col_axis_spacing} (between rows, PixelSpacing[0])"
        )

        if not np.isclose(
            spacing_between_rows,
            spacing_between_cols,
            rtol=self.PIXEL_SPACING_LOG_RTOL,
            atol=0.0,
        ):
            self._logger.info(f"Non-square pixels: PixelSpacing = [{spacing_between_rows}, {spacing_between_cols}]")

        return row_axis_spacing, col_axis_spacing

    def verify_affine_invariants(self, series, rtol: Optional[float] = None, raise_on_failure: bool = False) -> Dict:
        """Re-derives the affine's geometry from the first SOP instance and checks it.

        This is the assertion the test matrix should call in every case. It reads Image
        Orientation Patient, Pixel Spacing and Image Position Patient afresh rather than reusing
        the values compute_affine_transform worked from, so a bug in that method's bookkeeping
        shows up as a failed check rather than being reproduced on both sides.

        Checks:
            1. |det(m[:3,:3])| == row_axis_spacing * col_axis_spacing * depth_pixel_spacing.
               Catches a scalar/vector disagreement. Symmetric in the two in-plane spacings, so
               it does NOT catch a transposed pairing.
            2. n . m[:3,2] == depth_pixel_spacing. The scalar really is the projection of the
               vector, shear included.
            3. |m[:3,0]| == |row cosine| * the spacing paired with it. The cosine's own length is
               included because nothing is rescaled.
            4. |m[:3,1]| == |column cosine| * the spacing paired with it.
               Together, 3 and 4 are the only checks that see a transposed pairing, and they only
               see it on non-square pixels. This is what the non-square fixture exists for.
            5. m[:3,0] and m[:3,1], normalized, equal the normalized row and column cosines.
               Normalized on both sides because comparing a direction is scale-free; this check
               says nothing about length, which is 3 and 4's job.
            6. m[:3,3] == Image Position Patient of the first slice.
            7. The NIFTI affine is the DICOM affine with rows 0 and 1 negated.

        Args:
            series: An instance of DICOMSeries with the affines already computed.
            rtol: Relative tolerance; defaults to DETERMINANT_RTOL.
            raise_on_failure: If True, raise SliceGeometryError listing every failed check.
                              Default False: failures are logged, which is what the operator
                              wants at runtime. Tests should pass True.

        Returns:
            A dict of the computed quantities plus "failures", a list of strings.
        """
        tol = self.DETERMINANT_RTOL if rtol is None else float(rtol)
        failures: List[str] = []

        m1 = getattr(series, self.ATTRIBUTE_DICOM_AFFINE, None)
        m2 = getattr(series, self.ATTRIBUTE_NIFTI_AFFINE, None)
        if m1 is None:
            self._logger.debug("No DICOM affine on the series; invariant check skipped")
            return {"failures": ["no dicom_affine_transform on the series"]}

        s_1 = series._sop_instances[0]
        iop = self._get_tag_value(s_1, (0x0020, 0x0037))
        pixel_spacing = self._get_tag_value(s_1, (0x0028, 0x0030))
        ip1 = self._get_tag_value(s_1, (0x0020, 0x0032))
        depth = getattr(series, "depth_pixel_spacing", None)

        preconditions: List[str] = []
        if iop is None or len(iop) < 6:
            preconditions.append("Image Orientation Patient (0020,0037) is missing or malformed on the first slice")
        if pixel_spacing is None or len(pixel_spacing) < 2:
            preconditions.append("Pixel Spacing (0028,0030) is missing or malformed on the first slice")
        if ip1 is None or len(ip1) < 3:
            preconditions.append("Image Position Patient (0020,0032) is missing or malformed on the first slice")
        if depth is None:
            preconditions.append("series.depth_pixel_spacing is not set")
        if preconditions:
            message = "Affine invariant check could not run:\n  " + "\n  ".join(preconditions)
            if raise_on_failure:
                raise SliceGeometryError(message)
            self._logger.warning(message)
            return {"failures": preconditions}
        # The preconditions above return or raise whenever one of these is None; this only narrows the types
        assert iop is not None and pixel_spacing is not None and ip1 is not None and depth is not None

        row_cosine = np.asarray(iop[0:3], dtype=float)
        col_cosine = np.asarray(iop[3:6], dtype=float)
        normal = np.cross(row_cosine, col_cosine)
        row_norm = float(np.linalg.norm(row_cosine))
        col_norm = float(np.linalg.norm(col_cosine))
        row_unit = row_cosine / row_norm if row_norm else row_cosine
        col_unit = col_cosine / col_norm if col_norm else col_cosine

        # log=False: compute_affine_transform has already reported the pairing for this series
        # The verifier re-derives it independently, but reporting it twice adds no information
        row_axis_spacing, col_axis_spacing = self._pair_pixel_spacing(pixel_spacing, log=False)

        linear = np.asarray(m1)[:3, :3]
        det = abs(float(np.linalg.det(linear)))
        expected_det = row_axis_spacing * col_axis_spacing * float(depth)

        col0, col1, col2 = linear[:, 0], linear[:, 1], linear[:, 2]
        norm0, norm1 = float(np.linalg.norm(col0)), float(np.linalg.norm(col1))
        projected_depth = float(np.dot(normal, col2))

        def _check(name: str, actual: float, expected: float) -> None:
            ok = bool(np.isclose(actual, expected, rtol=tol, atol=0.0))
            rel = abs(actual - expected) / abs(expected) if expected else float("nan")
            self._logger.debug(
                f"Invariant {name}: {actual:.12f} vs {expected:.12f} (rel {rel:.3e}) {'OK' if ok else 'FAIL'}"
            )
            if not ok:
                failures.append(f"{name}: {actual:.12f} != {expected:.12f} (relative {rel:.3e})")

        _check("1 |det| == vr*vc*depth", det, expected_det)
        _check("2 n.col2 == depth", projected_depth, float(depth))
        _check("3 |col0| == |r| * row-cosine spacing", norm0, row_norm * row_axis_spacing)
        _check("4 |col1| == |c| * col-cosine spacing", norm1, col_norm * col_axis_spacing)

        for label, column, norm, cosine in (
            ("5a col0 direction", col0, norm0, row_unit),
            ("5b col1 direction", col1, norm1, col_unit),
        ):
            deviation = float(np.max(np.abs(column / norm - cosine))) if norm else float("inf")
            ok = deviation <= self.ORIENTATION_LOG_ATOL
            self._logger.debug(f"Invariant {label}: max deviation {deviation:.3e} {'OK' if ok else 'FAIL'}")
            if not ok:
                failures.append(f"{label}: max component deviation {deviation:.3e}")

        origin_deviation = float(np.max(np.abs(np.asarray(m1)[:3, 3] - np.asarray(ip1[:3], dtype=float))))
        self._logger.debug(f"Invariant 6 origin == IPP(first): max deviation {origin_deviation:.3e}")
        if origin_deviation > 0.0:
            failures.append(f"6 origin != IPP of the first slice: max deviation {origin_deviation:.3e}")

        if m2 is not None:
            expected_m2 = np.asarray(m1).copy()
            expected_m2[0, :] *= -1.0
            expected_m2[1, :] *= -1.0
            nifti_deviation = float(np.max(np.abs(np.asarray(m2) - expected_m2)))
            self._logger.debug(f"Invariant 7 NIFTI == LPS with rows 0,1 negated: max deviation {nifti_deviation:.3e}")
            if nifti_deviation > 0.0:
                failures.append(
                    f"7 NIFTI affine is not the LPS affine with rows 0 and 1 negated: {nifti_deviation:.3e}"
                )

        if failures:
            message = "Affine invariant check failed:\n  " + "\n  ".join(failures)
            if raise_on_failure:
                raise SliceGeometryError(message)
            self._logger.warning(message)
        else:
            self._logger.debug(
                f"All affine invariants hold to rtol {tol:.1e}: voxel volume {det:.9f} mm^3, "
                f"in-plane {norm0:.9f} x {norm1:.9f} mm, depth {float(depth):.9f} mm"
            )

        return {
            "determinant": det,
            "expected_determinant": expected_det,
            "depth_pixel_spacing": float(depth),
            "projected_depth": projected_depth,
            "row_axis_spacing": row_axis_spacing,
            "col_axis_spacing": col_axis_spacing,
            "col0_norm": norm0,
            "col1_norm": norm1,
            "failures": failures,
        }

    def create_metadata(self, series) -> Dict:
        """Collects all relevant metadata from the DICOM Series and creates a dictionary.

        Args:
            series: An instance of DICOMSeries.

        Returns:
            An instance of a dictionary containing metadata for the volumetric image.
        """

        # Set metadata with series properties that are not None.
        metadata = {}
        if series:
            metadata = self._get_instance_properties(series)
        return metadata

    @staticmethod
    def _get_instance_properties(obj: object, not_none: bool = True) -> Dict:
        prop_dict = {}
        if obj:
            for attribute in [x for x in type(obj).__dict__ if isinstance(type(obj).__dict__[x], property)]:
                attr_val = getattr(obj, attribute, None)
                if not_none:
                    if attr_val is not None:
                        prop_dict[attribute] = attr_val
                else:
                    prop_dict[attribute] = attr_val

        return prop_dict


def test():
    from pathlib import Path

    from monai.deploy.operators.dicom_data_loader_operator import DICOMDataLoaderOperator
    from monai.deploy.operators.dicom_series_selector_operator import DICOMSeriesSelectorOperator

    current_file_dir = Path(__file__).parent.resolve()
    data_path = current_file_dir.joinpath("../../../inputs/spleen_ct/dcm").absolute()

    fragment = Fragment()
    loader = DICOMDataLoaderOperator(fragment, name="loader_op")
    series_selector = DICOMSeriesSelectorOperator(fragment, name="selector_op")
    vol_op = DICOMSeriesToVolumeOperator(fragment, name="series_to_vol_op")

    study_list = loader.load_data_to_studies(data_path)
    study_selected_series_list = series_selector.filter(None, study_list)
    image = vol_op.convert_to_image(study_selected_series_list)

    print(f"Image NumPy array shape (index order DHW): {image.asnumpy().shape}")
    for k, v in image.metadata().items():
        print(f"{(k)}: {(v)}")


if __name__ == "__main__":
    test()
