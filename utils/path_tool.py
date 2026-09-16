"""
为整个工程提供统一的绝对路径
"""
import os

def get_project_root() ->str:
    """
    获取工程所在的根目录
    :return:字符串根目录
    """
    #获取当前文件的绝对路径，__file__代表当前文件
    current_file = os.path.abspath(__file__)
    #往上跳一级，获取当前文件所在文件夹的目录
    current_dir = os.path.dirname(current_file)
    # 往上跳一级，获取当前文件夹所在文件夹（project）的目录
    current_root = os.path.dirname(current_dir)
    return current_root


def get_abs_path(relative_path: str) -> str:
    """
    传递相对路径，得到绝对路径
    :param relative_path: 相对路径
    :return: 绝对路径
    """
    project_root = get_project_root()
    return os.path.join(project_root, relative_path)


if __name__ == '__main__':
    print(get_abs_path("config/config.txt"))