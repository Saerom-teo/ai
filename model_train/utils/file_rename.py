import os
import shutil
import re

def rename_files_in_folder(target_folder):
    # 정규 표현식을 사용하여 한글을 찾습니다.
    korean_pattern = re.compile('[가-힣]+')

    for root, dirs, files in os.walk(target_folder):
        for file_name in files:
            # 파일 이름에서 한글을 "ko"로 대체
            new_file_name = re.sub(korean_pattern, 'ko', file_name)
            
            if new_file_name != file_name:
                # 파일의 전체 경로 생성
                old_file_path = os.path.join(root, file_name)
                new_file_path = os.path.join(root, new_file_name)
                
                # 파일 이름 변경
                shutil.move(old_file_path, new_file_path)
                print(f'Renamed: {old_file_path} -> {new_file_path}')

# 사용 예시
# target_folder = 'your_target_folder'  # 변경하려는 대상 폴더의 경로
target_folder = 'E:/temp/recyclables_origin'
rename_files_in_folder(target_folder)
