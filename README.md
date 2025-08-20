# Create a properly formatted README.md
cat > README.md << 'EOL'
---
title: Kiswahili Chatbot
emoji: "🤖"
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.0.0"
app_file: app.py
pinned: false
---

# Kiswahili Chatbot 🤖

Mazungumzo ya AI kwa lugha ya Kiswahili. Wasiliana na msaidizi wako wa AI kwa lugha ya Kiswahili.

## Maelezo

Hii ni interface ya mazungumzo inayotumia modeli ya AI kutoka Hugging Face kuzungumza kwa Kiswahili.

## Matumizi

1. Andika ujumbe wako kwa Kiswahili katika kisanduku cha maandishi
2. Bonyeza kitufe cha "Tuma" au ubonyeze Enter
3. Subiri majibu kutoka kwa AI

## Sifa

- Inazungumza Kiswahili kikamilifu
- Interface rahisi na ya kisasa
- Inaweza kubadilika kwa hiari (temperature)
- Inatoa majibu ya haraka na ya haraka

## Teknolojia

- Gradio kwa interface
- Hugging Face Inference API
- Modeli: HuggingFaceH4/zephyr-7b-beta
EOL
