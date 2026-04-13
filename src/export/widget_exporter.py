"""
Widget Exporter — Generates embeddable HTML/JS chat widget.
"""
import json
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import EXPORTS_DIR


class WidgetExporter:
    """Exports a chatbot as an embeddable web widget."""

    def export(self, bot_config: dict, include_model: bool = False) -> Path:
        bot_name = bot_config.get("name", "Social Good Bot")
        bot_id = bot_config["bot_id"]
        export_dir = EXPORTS_DIR / f"{bot_id}_widget"
        export_dir.mkdir(parents=True, exist_ok=True)

        # Generate widget HTML/JS
        widget_html = f"""<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{bot_name}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        .sgb-widget {{
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 380px;
            max-height: 600px;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            z-index: 10000;
            border-radius: 16px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.15);
            overflow: hidden;
            display: flex;
            flex-direction: column;
            background: #fff;
            transition: all 0.3s ease;
        }}

        .sgb-widget.collapsed {{
            width: 64px;
            height: 64px;
            border-radius: 50%;
            cursor: pointer;
        }}

        .sgb-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 16px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .sgb-header h3 {{
            font-size: 16px;
            font-weight: 600;
        }}

        .sgb-header .badge {{
            background: rgba(255,255,255,0.2);
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 10px;
        }}

        .sgb-messages {{
            flex: 1;
            overflow-y: auto;
            padding: 16px;
            max-height: 400px;
            background: #f8f9fa;
        }}

        .sgb-message {{
            margin-bottom: 12px;
            display: flex;
            flex-direction: column;
        }}

        .sgb-message.user {{
            align-items: flex-end;
        }}

        .sgb-message.bot {{
            align-items: flex-start;
        }}

        .sgb-bubble {{
            max-width: 85%;
            padding: 10px 14px;
            border-radius: 12px;
            font-size: 14px;
            line-height: 1.5;
            word-wrap: break-word;
        }}

        .sgb-message.user .sgb-bubble {{
            background: #667eea;
            color: white;
            border-bottom-right-radius: 4px;
        }}

        .sgb-message.bot .sgb-bubble {{
            background: white;
            color: #333;
            border-bottom-left-radius: 4px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        }}

        .sgb-input-area {{
            padding: 12px 16px;
            border-top: 1px solid #e9ecef;
            display: flex;
            gap: 8px;
            background: white;
        }}

        .sgb-input-area input {{
            flex: 1;
            padding: 10px 14px;
            border: 1px solid #dee2e6;
            border-radius: 24px;
            font-size: 14px;
            outline: none;
            transition: border-color 0.2s;
        }}

        .sgb-input-area input:focus {{
            border-color: #667eea;
        }}

        .sgb-input-area button {{
            background: #667eea;
            color: white;
            border: none;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            cursor: pointer;
            font-size: 18px;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: background 0.2s;
        }}

        .sgb-input-area button:hover {{
            background: #764ba2;
        }}

        .sgb-fab {{
            width: 64px;
            height: 64px;
            border-radius: 50%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            display: none;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            box-shadow: 0 4px 16px rgba(102, 126, 234, 0.4);
            position: fixed;
            bottom: 20px;
            right: 20px;
            z-index: 10001;
        }}

        .sgb-fab svg {{
            fill: white;
            width: 28px;
            height: 28px;
        }}

        .sgb-typing {{
            display: flex;
            gap: 4px;
            padding: 8px 14px;
        }}

        .sgb-typing span {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #adb5bd;
            animation: sgb-bounce 1.4s infinite both;
        }}

        .sgb-typing span:nth-child(2) {{ animation-delay: 0.16s; }}
        .sgb-typing span:nth-child(3) {{ animation-delay: 0.32s; }}

        @keyframes sgb-bounce {{
            0%, 80%, 100% {{ transform: scale(0); }}
            40% {{ transform: scale(1); }}
        }}

        .sgb-disclaimer {{
            font-size: 11px;
            color: #6c757d;
            padding: 4px 16px 8px;
            text-align: center;
            background: white;
        }}
    </style>
</head>
<body>

<div id="sgb-widget" class="sgb-widget">
    <div class="sgb-header">
        <div>
            <h3>{bot_name}</h3>
        </div>
        <span class="badge">Social Good AI</span>
    </div>
    <div id="sgb-messages" class="sgb-messages">
        <div class="sgb-message bot">
            <div class="sgb-bubble">
                Merhaba! Ben {bot_name}. Size nasıl yardımcı olabilirim? 🤝
            </div>
        </div>
    </div>
    <div class="sgb-input-area">
        <input type="text" id="sgb-input" placeholder="Mesajınızı yazın..." autocomplete="off">
        <button id="sgb-send" aria-label="Gönder">➤</button>
    </div>
    <div class="sgb-disclaimer">
        Bu bot topluma fayda amacıyla geliştirilmiştir.
    </div>
</div>

<script>
(function() {{
    const API_URL = window.SGB_API_URL || 'http://localhost:8000/api/chat/send';
    const BOT_ID = '{bot_id}';

    const messagesEl = document.getElementById('sgb-messages');
    const inputEl = document.getElementById('sgb-input');
    const sendBtn = document.getElementById('sgb-send');

    function addMessage(content, isUser) {{
        const msgDiv = document.createElement('div');
        msgDiv.className = `sgb-message ${{isUser ? 'user' : 'bot'}}`;
        const bubble = document.createElement('div');
        bubble.className = 'sgb-bubble';
        bubble.textContent = content;
        msgDiv.appendChild(bubble);
        messagesEl.appendChild(msgDiv);
        messagesEl.scrollTop = messagesEl.scrollHeight;
    }}

    function showTyping() {{
        const typing = document.createElement('div');
        typing.id = 'sgb-typing';
        typing.className = 'sgb-message bot';
        typing.innerHTML = '<div class="sgb-typing"><span></span><span></span><span></span></div>';
        messagesEl.appendChild(typing);
        messagesEl.scrollTop = messagesEl.scrollHeight;
    }}

    function hideTyping() {{
        const el = document.getElementById('sgb-typing');
        if (el) el.remove();
    }}

    async function sendMessage() {{
        const text = inputEl.value.trim();
        if (!text) return;

        addMessage(text, true);
        inputEl.value = '';
        showTyping();

        try {{
            const res = await fetch(API_URL, {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{ bot_id: BOT_ID, message: text }})
            }});
            const data = await res.json();
            hideTyping();

            let response = data.response || 'Üzgünüm, bir hata oluştu.';
            if (data.disclaimer) {{
                response += '\\n\\n⚠️ ' + data.disclaimer;
            }}
            addMessage(response, false);
        }} catch (err) {{
            hideTyping();
            addMessage('Bağlantı hatası. Lütfen tekrar deneyin.', false);
        }}
    }}

    sendBtn.addEventListener('click', sendMessage);
    inputEl.addEventListener('keypress', (e) => {{
        if (e.key === 'Enter') sendMessage();
    }});
}})();
</script>
</body>
</html>"""

        # Embed script for external websites
        embed_script = f"""<!-- Social Good Bot Widget — {bot_name} -->
<script>
    window.SGB_API_URL = 'YOUR_API_URL_HERE/api/chat/send';
</script>
<script src="https://YOUR_CDN/sgb-widget.js"></script>
<!-- Or self-host the widget HTML and include via iframe: -->
<!-- <iframe src="widget.html" style="border:none;position:fixed;bottom:0;right:0;width:400px;height:620px;z-index:9999;"></iframe> -->
"""

        # Write files
        (export_dir / "widget.html").write_text(widget_html, encoding="utf-8")
        (export_dir / "embed.html").write_text(embed_script, encoding="utf-8")

        # Export metadata
        meta = {
            "bot_id": bot_id,
            "name": bot_name,
            "format": "widget",
            "exported_at": datetime.now().isoformat(),
            "path": str(export_dir),
        }
        with open(export_dir / "export_meta.json", "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        return export_dir
