import os
def CreateFolder(path):
    del_path_space = path.strip()
    del_path_tail = del_path_space.rstrip('\\')
    is_exists = os.path.exists(del_path_tail)
    if not is_exists:
        os.makedirs(del_path_tail)
        return True
    else:
        return False

