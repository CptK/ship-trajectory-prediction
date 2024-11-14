from typing import Literal
from abc import ABC, abstractmethod
import pandas as pd
from multiprocessing import cpu_count
from shapely.geometry import Polygon

from prediction.preprocessing.outlier_detection import remove_outliers_parallel
from prediction.data.filter_data import filter_by_travelled_distance, filter_data
from prediction.preprocessing.ais_status_segmentation import segment_by_status


class PipelineComponent(ABC):
    def __init__(self, name: str):
        self._name = name

    @abstractmethod
    def process(df: pd.DataFrame) -> pd.DataFrame:
        pass


class Pipeline:
    def __init__(self, components: list[PipelineComponent], verbose: bool = True):
        self.components = components
        self.verbose = verbose

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        for component in self.components:
            if self.verbose:
                print(f"Processing {component._name} ...")

            df = component.process(df)

            if self.verbose:
                print(f"Done processing {component._name}. The dataframe has {len(df)} rows.")
        return df
    

class OutlierDetection(PipelineComponent):
    def __init__(
        self,
        threshold_partition_sog: float = 5.0,
        threshold_partition_distance: float = 50.0,
        threshold_association_sog: float = 15.0,
        threshold_association_distance: float = 50.0,
        threshold_completeness: int = 100,
        additional_filter_columns: list[str] = [],
        verbose: bool = True,
        n_processes: int = -1,
    ):
        super().__init__("Outlier Detection")
        self.threshold_partition_sog = threshold_partition_sog
        self.threshold_partition_distance = threshold_partition_distance
        self.threshold_association_sog = threshold_association_sog
        self.threshold_association_distance = threshold_association_distance
        self.threshold_completeness = threshold_completeness
        self.additional_filter_columns = additional_filter_columns
        self.verbose = verbose
        self.n_processes = n_processes if n_processes != -1 else max(cpu_count() - 1, 1)

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        return remove_outliers_parallel(
            df=df,
            threshold_partition_sog=self.threshold_partition_sog,
            threshold_partition_distance=self.threshold_partition_distance,
            threshold_association_sog=self.threshold_association_sog,
            threshold_association_distance=self.threshold_association_distance,
            threshold_completeness=self.threshold_completeness,
            additional_filter_columns=self.additional_filter_columns,
            verbose=self.verbose,
            n_processes=self.n_processes,
        )


class AreaOfInterestFilter(PipelineComponent):
    def __init__(self, area_of_interest: Polygon, only_inside: bool = True):
        super().__init__("Area of Interest Filter")
        self.area_of_interest = area_of_interest
        self.only_inside = only_inside

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        return filter_data(df, self.area_of_interest, self.only_inside)


class TrajectorySizeFilter(PipelineComponent):
    def __init__(self, min_distance: float, max_distance: float, dist_col: str | None, method: Literal["exact", "approximate"] = "approximate", n_processes: int = -1):
        super().__init__("Trajectory Size Filter")
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.dist_col = dist_col
        self.method = method
        self.n_processes = n_processes if n_processes != -1 else max(cpu_count() - 1, 1)

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        return filter_by_travelled_distance(
            data=df,
            min_dist=self.min_distance,
            max_dist=self.max_distance,
            dist_col=self.dist_col,
            method=self.method,
            n_processes=self.n_processes,
        )


class TrajectoryStatusSegmentation(PipelineComponent):
    def __init__(
        self,
        status_column: str,
        default_status: int,
        split_statuses: list[int],
        min_segment_length: int | None = None,
        additional_list_columns: list[str] = ["geometry", "orientations", "velocities", "timestamps"],
        point_count_column: str | None = "point_count",
        n_processes: int | None = None,    
    ):
        super().__init__("Trajectory Status Segmentation")
        self.status_column = status_column
        self.default_status = default_status
        self.split_statuses = split_statuses
        self.min_segment_length = min_segment_length
        self.additional_list_columns = additional_list_columns
        self.point_count_column = point_count_column
        self.n_processes = n_processes

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        return segment_by_status(
            df=df,
            status_column=self.status_column,
            default_status=self.default_status,
            split_statuses=self.split_statuses,
            min_segment_length=self.min_segment_length,
            additional_list_columns=self.additional_list_columns,
            point_count_column=self.point_count_column,
            n_processes=self.n_processes,
        )
