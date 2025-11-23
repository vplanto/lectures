import os
import re
import shutil

# --- КОНФІГУРАЦІЯ ---

# Патерни для технічних даних (IP, DNS, MAC, Token)
REGEX_PATTERNS = [
    (r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '10.X.X.X'), # IPv4 masking
    (r'ip-\d+-\d+-\d+-\d+', 'ip-10-X-X-X'),                  # AWS Internal DNS
    (r'(?<=Bearer\s)[a-zA-Z0-9\-\._~\+\/]+=*', '<REDACTED_TOKEN>'), # Auth Tokens
    (r'[0-9a-fA-F]{12}', 'XXXXXXXXXXXX'),                    # MAC addresses
]

# Словники замін для конкретних сайтів (Бізнес-дані)
# УВАГА: Реальні назви проектів та доменів замінені на зірочки (****) для безпеки.
# Цей код демонструє логіку обробки, а не реальні дані клієнтів.
SITE_CONFIG = {
    # --- Сайт 1 (Alpha) ---
    "*******_prod": {  # <--- Назва папки з сирими даними
        "file_prefix": "site-alpha",
        "mappings": {
            "*******": "site-alpha",        # Головна назва проекту
            "*****": "legacy-app",          # Назва продукту
            "**-****-*": "region-1",        # Регіон (напр. eu-west-1)
            "*******.****": "internal.domain", # Специфічний FQDN
            ".****": ".internal.domain",       # Кореневий домен
            "compute.internal": "cluster.local"
        }
    },
    
    # --- Сайт 2 (Beta) ---
    "********_raw": {
        "file_prefix": "site-beta",
        "mappings": {
            "********": "site-beta",
            "******": "beta-core",
            # Складні ідентифікатори інфраструктури
            "********.****": "beta-infra.internal.domain",
            "********": "beta-infra",
            ".****": ".internal.domain",
            "pg": "db-service",
            "**-****-*": "region-2",
            "compute.internal": "cluster.local"
        }
    },

    # --- Сайт 3 (Gamma) ---
    "*****_raw": {
        "file_prefix": "site-gamma",
        "mappings": {
            "*****": "site-gamma",
            "***": "gamma-core",
            "**********-****": "gamma-infra",
            "**********": "gamma-infra",
            ".****": ".internal.domain",
            "pg": "db-service",
            "**-****-*": "region-3",
            "compute.internal": "cluster.local"
        }
    }
}

# Шляхи
SOURCE_ROOT = "00_raw_sources"
TARGET_ROOT = "01_clean_pool"

def obfuscate_text(text, mappings):
    # 1. Dictionary Replacement
    # Сортуємо ключі за довжиною (від найдовшого), щоб уникнути часткових замін.
    # Це критично для коректної заміни FQDN (наприклад, замінити "app.domain.com" раніше ніж "domain.com")
    sorted_keys = sorted(mappings.keys(), key=len, reverse=True)
    
    for key in sorted_keys:
        # Використовуємо regex для case-insensitive заміни
        pattern = re.compile(re.escape(key), re.IGNORECASE)
        text = pattern.sub(mappings[key], text)
    
    # 2. Regex Sanitization (Технічні дані)
    for pattern, replacement in REGEX_PATTERNS:
        text = re.sub(pattern, replacement, text)
    return text

def process_files():
    print(f"🚀 Starting obfuscation process...")
    
    if not os.path.exists(TARGET_ROOT):
        os.makedirs(TARGET_ROOT)

    for site_folder, config in SITE_CONFIG.items():
        target_prefix = config["file_prefix"]
        mappings = config["mappings"]
        
        source_path = os.path.join(SOURCE_ROOT, site_folder)
        
        # Перевірка наявності папки (в демо-режимі може бути відсутня)
        if not os.path.exists(source_path):
            print(f"⚠️  Note: Source folder '{site_folder}' not found. (Expected in template mode)")
            continue
            
        print(f"📂 Processing site: {site_folder} -> Target Prefix: {target_prefix}")

        for root, dirs, files in os.walk(source_path):
            for file in files:
                if file.startswith('.'): continue

                input_file_path = os.path.join(root, file)
                
                try:
                    with open(input_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                except Exception as e:
                    print(f"❌ Error reading {file}: {e}")
                    continue

                # Обфускуємо контент
                clean_content = obfuscate_text(content, mappings)

                # Формуємо нове ім'я файлу
                new_filename = file
                # Застосовуємо ті ж самі маппінги до імені файлу
                sorted_keys = sorted(mappings.keys(), key=len, reverse=True)
                for k in sorted_keys:
                    new_filename = new_filename.replace(k, mappings[k])
                
                # Перевіряємо та додаємо префікс сайту для унікальності
                if not new_filename.startswith(target_prefix):
                    new_filename = f"{target_prefix}_{new_filename}"
                
                output_file_path = os.path.join(TARGET_ROOT, new_filename)

                try:
                    with open(output_file_path, 'w', encoding='utf-8') as f:
                        f.write(clean_content)
                    print(f"   ✅ Processed: {file} -> {new_filename}")
                except Exception as e:
                    print(f"❌ Error writing {new_filename}: {e}")

    print(f"✨ Done. Processed files are in '{TARGET_ROOT}'.")

if __name__ == "__main__":
    process_files()