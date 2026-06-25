"""修复 LoRA 的 adapter_config.json：删掉当前 peft 版本不认识的字段。

背景：adapter 在新版 peft（含 aLoRA 特性）下训练，config 里带了 alora_invocation_tokens
等新字段；降级 peft 后旧版 LoraConfig 不认识这些字段，加载报 TypeError。
LoRA 权重本身是标准格式、与版本无关，所以只需把多余字段过滤掉即可，无需重训。
"""

import dataclasses
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import STUDENT_LORA_DIR
from pipeline.logger import get_logger

log = get_logger("fix_adapter")


def main():
    from peft import LoraConfig

    cfg_path = Path(STUDENT_LORA_DIR) / "adapter_config.json"
    if not cfg_path.exists():
        log.error("找不到 %s", cfg_path)
        sys.exit(1)

    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    accepted = {f.name for f in dataclasses.fields(LoraConfig)}

    dropped = {k: v for k, v in data.items() if k not in accepted}
    clean = {k: v for k, v in data.items() if k in accepted}

    if not dropped:
        log.info("没有需要删除的字段，config 已兼容当前 peft。")
        return

    # 备份原文件
    backup = cfg_path.with_name("adapter_config.json.bak")
    backup.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    cfg_path.write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")

    log.info("已删除不兼容字段: %s", list(dropped.keys()))
    log.info("原 config 备份到: %s", backup)
    log.info("已写回兼容版本: %s", cfg_path)


if __name__ == "__main__":
    main()
