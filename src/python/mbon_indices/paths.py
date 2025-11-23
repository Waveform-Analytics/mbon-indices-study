from pathlib import Path


def detections_excel_path(root: Path, year: int, station: str) -> Path:
    name = f"Master_Manual_{station}_2h_{year}.xlsx"
    if year == 2018:
        return root / "data" / "raw" / "2018" / "detections" / name
    if year == 2021:
        return root / "data" / "raw" / "2021" / "detections" / name
    return root / "data" / "raw" / str(year) / "detections" / name


def environmental_temp_path(root: Path, year: int, station: str) -> Path:
    name = f"Master_{station}_Temp_{year}.xlsx"
    return root / "data" / "raw" / str(year) / "environmental" / name


def environmental_depth_path(root: Path, year: int, station: str) -> Path:
    name = f"Master_{station}_Depth_{year}.xlsx"
    return root / "data" / "raw" / str(year) / "environmental" / name


def spl_excel_path(root: Path, year: int, station: str) -> Path:
    name = f"Master_rmsSPL_{station}_1h_{year}.xlsx"
    return root / "data" / "raw" / str(year) / "rms_spl" / name


def det_metadata_map_path(root: Path) -> Path:
    return root / "data" / "raw" / "metadata" / "det_column_names.csv"