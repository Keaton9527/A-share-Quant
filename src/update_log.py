"""
日志更新辅助程序 - 用于向开发日志中添加新版本或新的修改记录
"""
import os
import sys
import argparse
from datetime import datetime


def add_new_version(version, description=""):
    """
    添加新版本到日志文件
    
    Args:
        version: 版本号（如"v0.3"）
        description: 版本描述
    """
    log_file = 'development_log.md'
    
    if not os.path.exists(log_file):
        print(f"错误：找不到日志文件 {log_file}")
        return False
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 准备新版本模板
        current_date = datetime.now().strftime('%Y-%m-%d')
        new_version_template = f"""
---

# {version} ({current_date})

{description if description else "## 新功能/修改"}

### 运行结果

*待回测完成后补充*

"""
        # 查找插入位置
        first_version_index = content.find("# v0.1")
        if first_version_index == -1:
            print("错误：日志文件格式不正确，找不到合适的插入位置")
            return False
        
        # 找到第一个横线分隔符的位置
        divider_index = content.find("---")
        if divider_index == -1 or divider_index > first_version_index:
            # 如果没有找到分隔符或分隔符在第一个版本之后，则在第一个版本之前插入
            updated_content = content[:first_version_index] + new_version_template + content[first_version_index:]
        else:
            # 否则在第一个分隔符之后插入
            updated_content = content[:divider_index + 3] + new_version_template + content[divider_index + 3:]
        
        # 写回文件
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        print(f"已添加新版本 {version} 到日志文件")
        return True
    
    except Exception as e:
        print(f"更新日志文件时出错: {e}")
        return False


def add_new_modification(title, description):
    """
    添加新的修改记录到当前版本
    
    Args:
        title: 修改标题
        description: 修改描述（可以是多行文本）
    """
    log_file = 'development_log.md'
    
    if not os.path.exists(log_file):
        print(f"错误：找不到日志文件 {log_file}")
        return False
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找最新版本的位置
        latest_version_index = content.find("# v")
        if latest_version_index == -1:
            print("错误：日志文件格式不正确，找不到版本标记")
            return False
        
        # 查找下一个版本或结果部分
        next_section_index = content.find("### 运行结果", latest_version_index)
        if next_section_index == -1:
            print("错误：日志文件格式不正确，找不到运行结果部分")
            return False
        
        # 准备新的修改模板
        new_modification_template = f"""
## {title}

{description}

"""
        # 插入新的修改记录
        updated_content = content[:next_section_index] + new_modification_template + content[next_section_index:]
        
        # 写回文件
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        print(f"已添加新的修改记录 '{title}' 到日志文件")
        return True
    
    except Exception as e:
        print(f"更新日志文件时出错: {e}")
        return False


def main():
    """主程序入口，处理命令行参数"""
    parser = argparse.ArgumentParser(description="更新开发日志文件")
    subparsers = parser.add_subparsers(dest="command", help="子命令")
    
    # 添加新版本的命令
    version_parser = subparsers.add_parser("version", help="添加新版本")
    version_parser.add_argument("version", help="版本号，如'v0.3'")
    version_parser.add_argument("-d", "--description", help="版本描述")
    
    # 添加新修改的命令
    modification_parser = subparsers.add_parser("modify", help="添加新的修改记录")
    modification_parser.add_argument("title", help="修改标题")
    modification_parser.add_argument("description", help="修改描述")
    
    args = parser.parse_args()
    
    if args.command == "version":
        add_new_version(args.version, args.description)
    elif args.command == "modify":
        add_new_modification(args.title, args.description)
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 