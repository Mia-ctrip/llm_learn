import json

# 读取数据
with open('model_usage.guide.txt', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 统计
models = {}
gpus = {}
combinations = {}

for record in data:
    if all(record.values()):  # 跳过空记录
        model = record['Qwen3-vl-32b']
        gpu = record['L20_2_96']
        
        models[model] = models.get(model, 0) + 1
        gpus[gpu] = gpus.get(gpu, 0) + 1
        combo = f"{model} + {gpu}"
        combinations[combo] = combinations.get(combo, 0) + 1

print("=" * 100)
print("真实部署数据统计分析")
print("=" * 100)
print()

print("📊 最常用的模型（TOP 15）：")
print("-" * 80)
sorted_models = sorted(models.items(), key=lambda x: x[1], reverse=True)[:15]
for i, (model, count) in enumerate(sorted_models, 1):
    print(f"{i:2}. {model:50} - 出现 {count:3} 次")

print()
print("📊 最常用的显卡（TOP 10）：")
print("-" * 80)
sorted_gpus = sorted(gpus.items(), key=lambda x: x[1], reverse=True)[:10]
for i, (gpu, count) in enumerate(sorted_gpus, 1):
    print(f"{i:2}. {gpu:30} - 出现 {count:3} 次")

print()
print("🔗 最常见的模型 + 显卡组合（TOP 10）：")
print("-" * 80)
sorted_combos = sorted(combinations.items(), key=lambda x: x[1], reverse=True)[:10]
for i, (combo, count) in enumerate(sorted_combos, 1):
    parts = combo.split(' + ')
    print(f"{i:2}. {parts[0]:50} + {parts[1]:20} ({count:3}x)")

print()
print("=" * 100)
print(f"总部署数：{len(data)} 条记录")
print(f"模型种类：{len(models)}")
print(f"显卡种类：{len(gpus)}")
