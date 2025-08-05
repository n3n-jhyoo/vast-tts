import os
from glob import glob
import csv

def load_and_sort_by_key_length(input_dir: str) -> list[tuple[str, str]]:
    """
    지정된 경로에서 모든 CSV 파일을 읽고, key 기준 길이로 내림차순 정렬된 리스트 반환
    """
    csv_files = glob(os.path.join(input_dir, "*.csv"), recursive=True)
    merged_dict = {}
    except_filename = "sorted_dict.csv"
    for file in csv_files:
        if file.split('/')[-1]==except_filename:
            continue
        with open(file, encoding="utf-8") as f:
            reader = csv.reader(f)
            # headers = next(reader)  # 첫 줄은 header
            for row in reader:
                if len(row) >= 2:
                    key, value = row[0].strip(), row[1].strip()
                    merged_dict[key] = value  # 중복 key는 마지막 값으로 덮음

    # key 길이 기준 내림차순 정렬
    sorted_items = sorted(merged_dict.items(), key=lambda x: len(x[0]), reverse=True)

    with open(os.path.join(input_dir, "sorted_dict.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(sorted_items)


if __name__ == "__main__":
    load_and_sort_by_key_length("/home/jihyun/workspace/gitlab/speech2/text2phoneme/text2phoneme/normalization/ko/data")