import pandas as pd


# 从xlsx加载真实用户数据
def load_real_users(file_path):
    '''从xlsx或csv文件加载真实用户数据，如果失败则直接报错'''
    try:
        # 检查文件类型
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)

        if df.empty:
            raise ValueError("用户数据文件为空")

        users = []

        for _, row in df.iterrows():
            # 提取大五人格特质
            traits = {
                "开放性": row["开放性"] if pd.notna(row["开放性"]) else "中",
                "尽责性": row["尽责性"] if pd.notna(row["尽责性"]) else "中",
                "外向性": row["外向性"] if pd.notna(row["外向性"]) else "中",
                "宜人性": row["宜人性"] if pd.notna(row["宜人性"]) else "中",
                "神经质": row["神经质"] if pd.notna(row["神经质"]) else "中"
            }

            # 创建用户数据字典
            user = {
                "id": str(row["用户id"]) if pd.notna(row["用户id"]) else str(len(users)),
                "name": row["用户名"] if pd.notna(row["用户名"]) else f"未命名用户{len(users)}",
                "description": row["自我描述"] if pd.notna(row["自我描述"]) else "",
                "education": row["教育背景"] if pd.notna(row["教育背景"]) else "未知",
                "traits": traits,
                "health_opinion": row["健康观点"] if pd.notna(row["健康观点"]) else ""
            }
            users.append(user)

        if not users:
            raise ValueError("没有从文件中读取到有效用户数据")

        print(f"成功加载真实用户数据")
        return users
    except Exception as e:
        print(f"加载真实用户数据失败: {e}")
        raise RuntimeError(f"必须提供有效的用户数据文件且包含足够的用户数据。错误: {e}")
