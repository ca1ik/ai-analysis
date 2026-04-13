# Social Good Chatbot Platform

Topluma faydalı chatbot üretim, eğitim ve dağıtım platformu.

## Mimari

```
├── config/          # Konfigürasyon (model, eğitim, güvenlik)
├── src/
│   ├── core/        # Motor: Model yükleme, inference, güvenlik filtreleri
│   ├── training/    # QLoRA eğitim pipeline'ı
│   ├── generator/   # Bot fabrikası ve şablonlar
│   ├── export/      # Docker, Widget, API export
│   ├── api/         # FastAPI endpoint'leri
│   └── utils/       # GPU manager, logger
├── data/
│   ├── training_sets/  # Eğitim verileri (TR/EN)
│   ├── models/         # Fine-tune edilmiş modeller
│   └── processed/      # İşlenmiş veri setleri
└── exports/         # Dışa aktarılmış chatbot paketleri
```

## Hızlı Başlangıç

```bash
# 1. Setup
setup.bat

# 2. Başlat
start.bat

# 3. API Docs
# http://localhost:8000/docs
```

## API Kullanımı

### Bot Oluştur (Şablondan)
```bash
curl -X POST http://localhost:8000/api/bots/create \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Ruh Sağlığı Botu",
    "template": "mental_health_tr"
  }'
```

### Eğitim Başlat
```bash
# 1. Veri yükle
curl -X POST http://localhost:8000/api/training/upload-dataset \
  -F "file=@my_data.jsonl" -F "name=custom_data"

# 2. Eğitimi başlat
curl -X POST http://localhost:8000/api/training/train \
  -H "Content-Type: application/json" \
  -d '{
    "bot_name": "my_bot",
    "system_prompt": "Sen yardımsever bir asistansın.",
    "social_goal": "Toplum sağlığı",
    "dataset_name": "custom_data"
  }'
```

### Sohbet Et
```bash
curl -X POST http://localhost:8000/api/chat/send \
  -H "Content-Type: application/json" \
  -d '{"bot_id": "BOT_ID", "message": "Merhaba!"}'
```

### Dışa Aktar
```bash
curl -X POST http://localhost:8000/api/export/package \
  -H "Content-Type: application/json" \
  -d '{"bot_id": "BOT_ID", "format": "widget"}'
```

## Desteklenen Şablonlar

| Şablon | Dil | Sosyal Hedef |
|--------|-----|-------------|
| `mental_health_tr` | TR | Ruh sağlığı farkındalığı |
| `mental_health_en` | EN | Mental health support |
| `education_tr` | TR | Eğitime eşit erişim |
| `education_en` | EN | Equal access to education |
| `environment_tr` | TR | Çevre koruma |
| `crisis_response_tr` | TR | Kriz yönetimi |
| `elderly_care_tr` | TR | Yaşlı bakımı |
| `accessibility_en` | EN | Erişilebilirlik |

## Export Formatları

- **Docker**: Tek başına çalışan container + API
- **Web Widget**: Web sitesine gömülebilir chat arayüzü
- **Standalone API**: Bağımsız FastAPI sunucu paketi

## Teknik Detaylar

- **Base Model**: LLaMA 3 8B Instruct
- **Fine-tuning**: QLoRA (4-bit NF4 quantization)
- **GPU**: RTX 5070 12GB optimize
- **Framework**: HuggingFace + PEFT + TRL
- **API**: FastAPI + Uvicorn

## Güvenlik

- Nefret söylemi / şiddet filtresi
- Kriz durumu tespiti ve yönlendirme
- Tıbbi/hukuki sorumluluk uyarıları
- Yanlış bilgi yayılma engelleyici
- Dil tespiti (TR/EN otomatik)

## Eğitim Verisi Formatı

```jsonl
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Desteklenen formatlar: JSONL, JSON, CSV
Minimum: 10 örnek | Önerilen: 500+ örnek
