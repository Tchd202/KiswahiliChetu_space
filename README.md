# If emoji is still problematic, remove it completely
cat > README.md << 'EOF'
---
title: "Kiswahili Chatbot"
colorFrom: "blue"
colorTo: "green"
sdk: "gradio"
sdk_version: "4.0.0"
app_file: "app.py"
pinned: false
---

# Kiswahili Chatbot

## Karibu kwenye Mazingira ya Mazungumzo ya Kiswahili

### Huduma za Mazungumzo
- **Mazungumzo ya Kiswahili** - Wasiliana na AI kwa lugha ya Kiswahili
- **Tafsiri** - Pata tafsiri bora kati ya lugha mbalimbali
- **Msaada wa Lugha** - Pata mwongozo wa sarufi na matumizi ya Kiswahili
EOF
