#!/usr/bin/env python3
"""
🌐 爬取蘑菇相关延展性话题的维基百科内容
扩展知识库：菌类学、烹饪、医学、生态学等
"""

import wikipediaapi
import json
import time
import os

# ================= 配置 =================
OUTPUT_FILE = "data/raw_data_source/related_topics_wiki_data.json"
SLEEP_BETWEEN_REQUESTS = 1  # 避免被封IP

# 相关主题列表（维基百科页面名称）
RELATED_TOPICS = [
    # === 基础生物学 ===
    "Fungus",                      # 真菌
    "Mycelium",                    # 菌丝
    "Spore",                       # 孢子
    "Basidiomycota",               # 担子菌门
    "Ascomycota",                  # 子囊菌门
    "Fungal_life_cycle",           # 真菌生命周期

    # === 蘑菇相关 ===
    "Mushroom",                    # 蘑菇（总述）
    "Edible_mushroom",             # 食用菌
    "Mushroom_poisoning",          # 蘑菇中毒
    "Mushroom_hunting",            # 采蘑菇/觅菇
    "Medicinal_mushrooms",         # 药用蘑菇
    "Psychedelic_mushroom",        # 致幻蘑菇

    # === 栽培与产业 ===
    "Mushroom_cultivation",        # 蘑菇栽培
    "Fungiculture",                # 真菌培养
    "Mushroom_spawn",              # 菌种

    # === 烹饪与应用 ===
    "Mushroom_soup",               # 蘑菇汤
    "Shiitake",                    # 香菇（常见食用菌）
    "Button_mushroom",             # 双孢蘑菇
    "Oyster_mushroom",             # 平菇
    "Enoki_mushroom",              # 金针菇

    # === 生态与科学 ===
    "Mycology",                    # 真菌学
    "Mycorrhiza",                  # 菌根
    "Decomposer",                  # 分解者
    "Fungal_ecology",              # 真菌生态学

    # === 毒理学 ===
    "Amatoxin",                    # 鹅膏毒素
    "Muscimol",                    # 蝇蕈醇
    "Psilocybin",                  # 裸盖菇素
    "Mushroom_toxin",              # 蘑菇毒素

    # === 识别与安全 ===
    "Mushroom_identification",     # 蘑菇识别
    "Lookalike_mushroom",          # 相似蘑菇
    "Foraging",                    # 觅食
]

# ========================================

def fetch_wikipedia_content(wiki_wiki, topic):
    """
    爬取维基百科页面内容（使用 wikipedia-api）
    """
    try:
        # 获取页面
        page = wiki_wiki.page(topic)

        if not page.exists():
            return None

        # 提取标题
        title = page.title

        # 提取摘要（第一段）
        summary = page.summary[:1200] if page.summary else ""

        # 提取章节内容
        sections = {}

        def extract_sections(section, depth=0, max_depth=2):
            """递归提取章节内容（只取前2层）"""
            if depth >= max_depth:
                return

            for s in section.sections:
                # 跳过常见的无用章节
                if s.title.lower() in ['references', 'external links', 'see also', 'notes', 'bibliography']:
                    continue

                # 保存章节内容
                if s.text and len(s.text) > 100:
                    sections[s.title] = s.text[:1200]  # 限制长度

                    # 只保存前5个章节
                    if len(sections) >= 6:
                        return

                # 递归提取子章节
                extract_sections(s, depth + 1, max_depth)

                if len(sections) >= 5:
                    return

        # 从根章节开始提取
        extract_sections(page, depth=0)

        return {
            "topic": title,
            "wiki_url": page.fullurl,
            "summary": summary,
            "sections": sections
        }

    except Exception as e:
        print(f"   ❌ 失败: {e}")
        return None

def main():
    print("=" * 80)
    print("🌐 爬取蘑菇相关延展性话题")
    print("=" * 80)
    print(f"\n📋 计划爬取 {len(RELATED_TOPICS)} 个主题\n")

    # 初始化 Wikipedia API
    wiki_wiki = wikipediaapi.Wikipedia(
        language='en',
        user_agent='MushroomKnowledgeBot/1.0 (educational purposes)'
    )

    all_data = []
    success_count = 0

    for idx, topic in enumerate(RELATED_TOPICS, 1):
        print(f"[{idx}/{len(RELATED_TOPICS)}] 爬取: {topic.replace('_', ' ')}", end=" ", flush=True)

        data = fetch_wikipedia_content(wiki_wiki, topic)

        if data:
            all_data.append(data)
            success_count += 1
            print(f"✅ ({len(data['summary'])} 字符)")
        else:
            print("⚠️  跳过")

        time.sleep(SLEEP_BETWEEN_REQUESTS)

    # 保存数据
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print(f"✅ 完成！成功爬取 {success_count}/{len(RELATED_TOPICS)} 个主题")
    print(f"📁 保存到: {OUTPUT_FILE}")
    print("=" * 80)

    # 预览
    print("\n📝 数据预览:")
    for i, item in enumerate(all_data[:3], 1):
        print(f"\n【{i}】{item['topic']}")
        print(f"   摘要: {item['summary'][:150]}...")
        print(f"   章节数: {len(item['sections'])}")

if __name__ == "__main__":
    main()
