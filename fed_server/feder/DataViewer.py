

from DataLoader import DataLoader as loader
import h5py

def print_h5_structure_attr(filepath):
    with h5py.File(filepath, "r") as f:
        print("文件结构:")
        def visit(name, obj):
            indent = "    " * name.count("/")
            if isinstance(obj, h5py.Group):
                print(f"{indent}组: {name}")
            elif isinstance(obj, h5py.Dataset):
                print(f"{indent}数据集: {name} (形状: {obj.shape}, 类型: {obj.dtype})")
        f.visititems(visit)

        print("\n全局属性:", dict(f.attrs))
        if "metadata" in f:
            print("metadata 组属性:", dict(f["metadata"].attrs))


def print_h5_metadata(filepath):
    with h5py.File(filepath, 'r') as f:
        if 'metadata' in f:
            metadata_group = f['metadata']
            print(f"metadata 组包含的对象: {list(metadata_group.keys())}")
        else:
            print("文件中没有 metadata 组")
def print_h5_structure(filepath):
    with h5py.File(filepath, 'r') as f:
        print("HDF5 文件结构:")
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"数据集: {name}, 形状: {obj.shape}, 数据类型: {obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"组: {name}")
        f.visititems(visit)

# 使用示例
if __name__ == '__main__':
    #filepath = "..\..\..\data\client_003_20250809_123254.h5"
    #filepath = "..\..\..\data\client_004_20250809_155237.h5"
    filepath = "..\..\..\data\client_004.h5"

    print_h5_metadata(filepath)
    print_h5_structure(filepath)
    print_h5_structure_attr(filepath)




