"""Step 4a: 从魔搭 ModelScope 下载 7B 学生模型权重到本地。"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import STUDENT_MODEL_ID, STUDENT_LOCAL_DIR
from pipeline.logger import get_logger

log = get_logger("step04_dl")


def main():
    from modelscope import snapshot_download

    log.info("下载 %s -> %s", STUDENT_MODEL_ID, STUDENT_LOCAL_DIR)
    path = snapshot_download(
        STUDENT_MODEL_ID,
        cache_dir=str(Path(STUDENT_LOCAL_DIR).parent),
        local_dir=STUDENT_LOCAL_DIR,
    )
    log.info("完成: %s", path)


if __name__ == "__main__":
    main()
