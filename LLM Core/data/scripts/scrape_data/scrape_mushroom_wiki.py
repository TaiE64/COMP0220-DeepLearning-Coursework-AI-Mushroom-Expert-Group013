import wikipediaapi
import json
import time

# 1. 定义要爬取的蘑菇列表（100种常见蘑菇）
mushroom_list = [
    # 有毒蘑菇
    "Amanita muscaria",       # 毒蝇伞
    "Amanita phalloides",     # 死亡帽
    "Amanita virosa",         # 毁灭天使
    "Galerina marginata",     # 边缘环柄菇
    "Conocybe filaris",       # 裸伞

    # 可食用蘑菇
    "Boletus edulis",         # 牛肝菌
    "Cantharellus cibarius",  # 鸡油菌
    "Pleurotus ostreatus",    # 平菇
    "Lentinula edodes",       # 香菇
    "Agaricus bisporus",      # 双孢蘑菇
    "Morchella esculenta",    # 羊肚菌
    "Tuber melanosporum",     # 黑松露
    "Lactarius deliciosus",   # 美味牛肝菌
    "Hericium erinaceus",     # 猴头菇

    # 药用蘑菇
    "Ganoderma lingzhi",      # 灵芝
    "Cordyceps sinensis",     # 冬虫夏草
    "Trametes versicolor",    # 云芝
    "Inonotus obliquus",      # 白桦茸

    # 致幻蘑菇
    "Psilocybe cubensis",     # 裸盖菇
    "Psilocybe semilanceata", # 半披针菌

    # 其他知名蘑菇
    "Amanita caesarea",       # 橙盖鹅膏
    "Coprinus comatus",       # 鸡腿菇
    "Flammulina velutipes",   # 金针菇
    "Tremella fuciformis",    # 银耳
    "Auricularia auricula-judae", # 木耳
    "Armillaria mellea",      # 蜜环菌
    "Clitocybe nebularis",    # 云斑蘑菇
    "Hygrocybe conica",       # 锥形湿伞
    "Lactarius torminosus",   # 绒毛乳菇
    "Russula emetica",        # 呕吐红菇
    "Suillus luteus",         # 黄褐乳牛肝菌
    "Tricholoma matsutake",   # 松茸
    "Volvariella volvacea",   # 草菇
    "Macrolepiota procera",   # 高大环柄菇
    "Lycoperdon perlatum",    # 马勃
]

# 2. 初始化 Wiki API
wiki_wiki = wikipediaapi.Wikipedia(
    user_agent='MushroomKnowledgeBot/1.0 (Educational purpose; contact: research@example.com)',
    language='en'
)

def get_page_data(topic):
    """从维基百科获取指定主题的数据 - 抓取所有章节"""
    page = wiki_wiki.page(topic)

    if not page.exists():
        print(f"❌ 页面不存在: {topic}")
        return None

    print(f"✅ 正在抓取: {topic}")

    # 提取基础信息
    data = {
        "topic": topic,
        "url": page.fullurl,
        "summary": page.summary,  # 完整摘要
        "full_text": page.text,   # 完整正文
        "sections": {}
    }

    # 递归提取所有章节（不过滤）
    def extract_sections(sections, prefix=""):
        for section in sections:
            # 构建层级章节路径
            section_path = f"{prefix}{section.title}" if prefix else section.title

            # 保存所有章节内容
            data["sections"][section_path] = section.text

            # 递归处理子章节
            if section.sections:
                extract_sections(section.sections, f"{section_path} > ")

    extract_sections(page.sections)

    return data

# 3. 主循环抓取
print("=" * 60)
print("开始抓取维基百科蘑菇知识")
print("=" * 60)

all_mushroom_knowledge = []
failed_topics = []

for i, mushroom in enumerate(mushroom_list, 1):
    print(f"[{i}/{len(mushroom_list)}] ", end="")

    info = get_page_data(mushroom)
    if info:
        all_mushroom_knowledge.append(info)
    else:
        failed_topics.append(mushroom)

    # 礼貌爬虫：每次暂停 1 秒
    time.sleep(1)

# 4. 保存为原始 JSON 数据
output_file = "data/raw_data_source/raw_mushroom_wiki_data.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(all_mushroom_knowledge, f, indent=4, ensure_ascii=False)

print("\n" + "=" * 60)
print(f"🎉 抓取完成！")
print(f"✅ 成功获取: {len(all_mushroom_knowledge)} 个条目")
print(f"❌ 失败条目: {len(failed_topics)} 个")
if failed_topics:
    print(f"   失败列表: {', '.join(failed_topics)}")
print(f"📁 数据已保存到: {output_file}")
print("=" * 60)
