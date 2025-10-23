import pandas as pd
from pathlib import Path
from typing import List, Union

def convert_csv_to_parquet(
    csv_dir: Union[str, Path],
    parquet_dir: Union[str, Path],
    create_dir: bool = True
) -> List[Path]:
    """
    Convert all CSV files in csv_dir to Parquet files in parquet_dir.

    Args:
        csv_dir: Directory containing CSV files
        parquet_dir: Directory to save Parquet files
        create_dir: Whether to create the output directory if it doesn't exist

    Returns:
        List of paths to created Parquet files
    """
    # Convert to Path objects
    csv_dir_path = Path(csv_dir)
    parquet_dir_path = Path(parquet_dir)

    # Check if CSV directory exists
    if not csv_dir_path.exists():
        raise FileNotFoundError(f"CSV directory does not exist: {csv_dir_path}")

    # Create parquet directory if it doesn't exist and create_dir is True
    if not parquet_dir_path.exists():
        if create_dir:
            parquet_dir_path.mkdir(parents=True, exist_ok=True)
        else:
            raise FileNotFoundError(f"Parquet directory does not exist: {parquet_dir_path}")

    # Get all CSV files
    csv_files = list(csv_dir_path.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {csv_dir_path}")
        return []

    created_files: List[Path] = []

    # Process each CSV file
    for csv_file in csv_files:
        try:
            # Read CSV
            df = pd.read_csv(csv_file)

            # Determine output path
            parquet_file = parquet_dir_path / f"{csv_file.stem}.parquet"

            # Save as Parquet
            df.to_parquet(parquet_file)

            created_files.append(parquet_file)
            print(f"Converted {csv_file} to {parquet_file}")

        except Exception as e:
            print(f"Error converting {csv_file}: {e}")

    return created_files


def main() -> None:
    """Main function to run the conversion."""
    csv_dir = Path("data/video_features/csv/")
    parquet_dir = Path("data/video_features/parquet/")

    try:
        converted_files = convert_csv_to_parquet(csv_dir, parquet_dir)
        print(f"Successfully converted {len(converted_files)} CSV files to Parquet format.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()