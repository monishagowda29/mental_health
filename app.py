from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from PIL import Image
import os, io, base64, logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MindScanApp")

from src.config import Config
from src.services.translation import TranslationService
from src.services.vision import GroqVisionService

# ═══════════════════════════════════════════════════
# FULL UI LANGUAGE STRINGS (UI changes, not just input)
# ═══════════════════════════════════════════════════
UI_STRINGS = {
    "en": {
        "flag": "🇬🇧", "name": "English",
        "tagline": "Multilingual BERT Mental Health Risk Prediction & Vision Analysis",
        "caption": "Mysore University · AI&DS · Enterprise Architecture V2.0",
        "status_bert": "BERT Model Ready",
        "status_vision": "Vision API Ready",
        "status_vision_off": "Vision API Offline",
        "tab_text": "📝 Text Analysis",
        "tab_image": "🖼️ Image Analysis",
        "tab_batch": "⚡ Batch Analysis",
        "input_label": "Enter your thoughts or describe how you feel:",
        "input_placeholder": "Type or paste text in English here...",
        "btn_analyze": "🔍 Analyze Now",
        "spinner_translate": "Translating securely offline...",
        "spinner_bert": "Running BERT inference...",
        "result_title": "Diagnosis Result",
        "confidence_title": "Confidence Breakdown",
        "disclaimer": "⚠️ Research prototype only. NOT a clinical tool. Always consult a qualified professional.",
        "img_upload": "Upload image (JPG, PNG, WEBP, BMP):",
        "btn_vision": "🔬 Analyze Image",
        "spinner_vision": "Analyzing via Groq Vision...",
        "batch_label": "Paste multiple entries (one per line):",
        "batch_placeholder": "Post 1 (English)\nPost 2 (any Indian language)\nPost 3...",
        "btn_batch": "⚡ Run Batch Analysis",
        "batch_summary": "Batch Summary",
        "total": "Total Records",
        "depression": "Depression Risk",
        "anxiety": "Anxiety Risk",
        "normal": "Normal",
        "download_csv": "⬇️ Download CSV",
        "download_txt": "⬇️ Download Report",
        "advice_normal": "No significant risk indicators detected. Maintain your healthy routine.",
        "advice_anxiety": "Anxiety patterns detected. Consider speaking with a mental health professional.",
        "advice_depression": "Depression patterns detected. Please reach out to a mental health professional.",
        "choose_lang": "Choose your language",
        "model_info": "Fine-tuned BERT · bert-base-uncased · Reddit Mental Health Dataset",
        "offline_badge": "🔒 Offline Translation",
    },
    "hi": {
        "flag": "🇮🇳", "name": "हिन्दी",
        "tagline": "बहुभाषी BERT मानसिक स्वास्थ्य जोखिम विश्लेषण",
        "caption": "मैसूर विश्वविद्यालय · AI&DS · एंटरप्राइज V2.0",
        "status_bert": "BERT मॉडल तैयार है",
        "status_vision": "विज़न API तैयार है",
        "status_vision_off": "विज़न API उपलब्ध नहीं",
        "tab_text": "📝 पाठ विश्लेषण",
        "tab_image": "🖼️ छवि विश्लेषण",
        "tab_batch": "⚡ बैच विश्लेषण",
        "input_label": "अपने विचार लिखें या अपनी भावना बताएं:",
        "input_placeholder": "यहाँ हिन्दी में टेक्स्ट टाइप करें...",
        "btn_analyze": "🔍 अभी विश्लेषण करें",
        "spinner_translate": "ऑफलाइन अनुवाद हो रहा है...",
        "spinner_bert": "BERT विश्लेषण चल रहा है...",
        "result_title": "निदान परिणाम",
        "confidence_title": "विश्वास स्तर",
        "disclaimer": "⚠️ शोध प्रोटोटाइप केवल। नैदानिक उपकरण नहीं। हमेशा किसी पेशेवर से परामर्श लें।",
        "img_upload": "छवि अपलोड करें (JPG, PNG, WEBP, BMP):",
        "btn_vision": "🔬 छवि विश्लेषण करें",
        "spinner_vision": "Groq Vision के माध्यम से विश्लेषण...",
        "batch_label": "एकाधिक प्रविष्टियाँ पेस्ट करें (प्रति पंक्ति एक):",
        "batch_placeholder": "पंक्ति 1 (हिन्दी)\nLine 2 (English)\nLínea 3...",
        "btn_batch": "⚡ बैच विश्लेषण चलाएं",
        "batch_summary": "बैच सारांश",
        "total": "कुल रिकॉर्ड",
        "depression": "अवसाद जोखिम",
        "anxiety": "चिंता जोखिम",
        "normal": "सामान्य",
        "download_csv": "⬇️ CSV डाउनलोड करें",
        "download_txt": "⬇️ रिपोर्ट डाउनलोड करें",
        "advice_normal": "कोई महत्वपूर्ण जोखिम नहीं मिला। अपनी स्वस्थ दिनचर्या बनाए रखें।",
        "advice_anxiety": "चिंता के संकेत मिले। किसी मानसिक स्वास्थ्य विशेषज्ञ से बात करें।",
        "advice_depression": "अवसाद के संकेत मिले। कृपया किसी मानसिक स्वास्थ्य विशेषज्ञ से संपर्क करें।",
        "choose_lang": "भाषा चुनें",
        "model_info": "Fine-tuned BERT · Reddit मानसिक स्वास्थ्य डेटासेट",
        "offline_badge": "🔒 ऑफलाइन अनुवाद",
    },
    "kn": {
        "flag": "🇮🇳", "name": "ಕನ್ನಡ",
        "tagline": "ಬಹುಭಾಷಾ BERT ಮಾನಸಿಕ ಆರೋಗ್ಯ ಅಪಾಯ ವಿಶ್ಲೇಷಣೆ",
        "caption": "ಮೈಸೂರು ವಿಶ್ವವಿದ್ಯಾಲಯ · AI&DS · ಎಂಟರ್‌ಪ್ರೈಸ್ V2.0",
        "status_bert": "BERT ಮಾದರಿ ಸಿದ್ಧವಾಗಿದೆ",
        "status_vision": "ವಿಷನ್ API ಸಿದ್ಧ",
        "status_vision_off": "ವಿಷನ್ API ಆಫ್‌ಲೈನ್",
        "tab_text": "📝 ಪಠ್ಯ ವಿಶ್ಲೇಷಣೆ",
        "tab_image": "🖼️ ಚಿತ್ರ ವಿಶ್ಲೇಷಣೆ",
        "tab_batch": "⚡ ಬ್ಯಾಚ್ ವಿಶ್ಲೇಷಣೆ",
        "input_label": "ನಿಮ್ಮ ಆಲೋಚನೆಗಳನ್ನು ಅಥವಾ ಭಾವನೆಯನ್ನು ನಮೂದಿಸಿ:",
        "input_placeholder": "ಇಲ್ಲಿ ಕನ್ನಡದಲ್ಲಿ ಟೈಪ್ ಮಾಡಿ...",
        "btn_analyze": "🔍 ಈಗ ವಿಶ್ಲೇಷಿಸಿ",
        "spinner_translate": "ಆಫ್‌ಲೈನ್‌ನಲ್ಲಿ ಅನುವಾದ ಮಾಡಲಾಗುತ್ತಿದೆ...",
        "spinner_bert": "BERT ವಿಶ್ಲೇಷಣೆ ನಡೆಯುತ್ತಿದೆ...",
        "result_title": "ನಿದಾನ ಫಲಿತಾಂಶ",
        "confidence_title": "ವಿಶ್ವಾಸ ಮಟ್ಟ",
        "disclaimer": "⚠️ ಸಂಶೋಧನಾ ಮಾದರಿ ಮಾತ್ರ. ವೈದ್ಯಕೀಯ ಸಾಧನವಲ್ಲ. ಯಾವಾಗಲೂ ತಜ್ಞರನ್ನು ಸಂಪರ್ಕಿಸಿ.",
        "img_upload": "ಚಿತ್ರ ಅಪ್‌ಲೋಡ್ ಮಾಡಿ (JPG, PNG, WEBP, BMP):",
        "btn_vision": "🔬 ಚಿತ್ರ ವಿಶ್ಲೇಷಿಸಿ",
        "spinner_vision": "Groq Vision ಮೂಲಕ ವಿಶ್ಲೇಷಿಸಲಾಗುತ್ತಿದೆ...",
        "batch_label": "ಬಹು ನಮೂದುಗಳನ್ನು ಅಂಟಿಸಿ (ಪ್ರತಿ ಸಾಲು ಒಂದು):",
        "batch_placeholder": "ಸಾಲು 1 (ಕನ್ನಡ)\nLine 2 (English)\nसाल 3 (Hindi)...",
        "btn_batch": "⚡ ಬ್ಯಾಚ್ ವಿಶ್ಲೇಷಣೆ ಚಲಾಯಿಸಿ",
        "batch_summary": "ಬ್ಯಾಚ್ ಸಾರಾಂಶ",
        "total": "ಒಟ್ಟು ದಾಖಲೆಗಳು",
        "depression": "ಖಿನ್ನತೆ ಅಪಾಯ",
        "anxiety": "ಆತಂಕ ಅಪಾಯ",
        "normal": "ಸಾಮಾನ್ಯ",
        "download_csv": "⬇️ CSV ಡೌನ್‌ಲೋಡ್",
        "download_txt": "⬇️ ವರದಿ ಡೌನ್‌ಲೋಡ್",
        "advice_normal": "ಯಾವುದೇ ಗಮನಾರ್ಹ ಅಪಾಯ ಕಂಡುಬಂದಿಲ್ಲ. ಆರೋಗ್ಯಕರ ಜೀವನಶೈಲಿ ಮುಂದುವರಿಸಿ.",
        "advice_anxiety": "ಆತಂಕದ ಚಿಹ್ನೆಗಳು ಕಂಡುಬಂದಿವೆ. ಮಾನಸಿಕ ಆರೋಗ್ಯ ತಜ್ಞರನ್ನು ಸಂಪರ್ಕಿಸಿ.",
        "advice_depression": "ಖಿನ್ನತೆಯ ಚಿಹ್ನೆಗಳು ಕಂಡುಬಂದಿವೆ. ದಯವಿಟ್ಟು ತಜ್ಞರನ್ನು ಸಂಪರ್ಕಿಸಿ.",
        "choose_lang": "ಭಾಷೆ ಆಯ್ಕೆ ಮಾಡಿ",
        "model_info": "Fine-tuned BERT · Reddit ಮಾನಸಿಕ ಆರೋಗ್ಯ ಡೇಟಾಸೆಟ್",
        "offline_badge": "🔒 ಆಫ್‌ಲೈನ್ ಅನುವಾದ",
    },
    "ta": {
        "flag": "🇮🇳", "name": "தமிழ்",
        "tagline": "பன்மொழி BERT மன நல ஆபத்து பகுப்பாய்வு",
        "caption": "மைசூர் பல்கலைக்கழகம் · AI&DS · Enterprise V2.0",
        "status_bert": "BERT மாதிரி தயார்",
        "status_vision": "Vision API தயார்",
        "status_vision_off": "Vision API இல்லை",
        "tab_text": "📝 உரை பகுப்பாய்வு",
        "tab_image": "🖼️ படம் பகுப்பாய்வு",
        "tab_batch": "⚡ தொகுதி பகுப்பாய்வு",
        "input_label": "உங்கள் எண்ணங்களை அல்லது உணர்வுகளை உள்ளிடவும்:",
        "input_placeholder": "இங்கே தமிழில் தட்டச்சு செய்யுங்கள்...",
        "btn_analyze": "🔍 இப்போது பகுப்பாய்வு செய்",
        "spinner_translate": "ஆஃப்லைனில் மொழிபெயர்க்கப்படுகிறது...",
        "spinner_bert": "BERT பகுப்பாய்வு இயங்குகிறது...",
        "result_title": "நோயறிதல் முடிவு",
        "confidence_title": "நம்பிக்கை அளவு",
        "disclaimer": "⚠️ ஆராய்ச்சி முன்மாதிரி மட்டும். மருத்துவ கருவி அல்ல.",
        "img_upload": "படத்தை பதிவேற்றவும் (JPG, PNG, WEBP, BMP):",
        "btn_vision": "🔬 படத்தை பகுப்பாய்வு செய்",
        "spinner_vision": "Groq Vision மூலம் பகுப்பாய்வு...",
        "batch_label": "பல உள்ளீடுகளை ஒட்டவும் (வரிக்கு ஒன்று):",
        "batch_placeholder": "வரி 1 (தமிழ்)\nLine 2 (English)\nसाल 3...",
        "btn_batch": "⚡ தொகுதி பகுப்பாய்வை இயக்கு",
        "batch_summary": "தொகுதி சுருக்கம்",
        "total": "மொத்த பதிவுகள்",
        "depression": "மனச்சோர்வு ஆபத்து",
        "anxiety": "பதற்றம் ஆபத்து",
        "normal": "சாதாரண",
        "download_csv": "⬇️ CSV பதிவிறக்கம்",
        "download_txt": "⬇️ அறிக்கை பதிவிறக்கம்",
        "advice_normal": "குறிப்பிடத்தக்க ஆபத்து இல்லை. ஆரோக்கியமான வாழ்க்கையை தொடரவும்.",
        "advice_anxiety": "பதற்ற அறிகுறிகள் உள்ளன. மன நல நிபுணரிடம் பேசவும்.",
        "advice_depression": "மனச்சோர்வு அறிகுறிகள் உள்ளன. மன நல நிபுணரை தொடர்பு கொள்ளவும்.",
        "choose_lang": "மொழி தேர்ந்தெடுக்கவும்",
        "model_info": "Fine-tuned BERT · Reddit மன நல Dataset",
        "offline_badge": "🔒 ஆஃப்லைன் மொழிபெயர்ப்பு",
    },
    "te": {
        "flag": "🇮🇳", "name": "తెలుగు",
        "tagline": "బహుభాషా BERT మానసిక ఆరోగ్య ప్రమాద విశ్లేషణ",
        "caption": "మైసూరు విశ్వవిద్యాలయం · AI&DS · Enterprise V2.0",
        "status_bert": "BERT మోడల్ సిద్ధంగా ఉంది",
        "status_vision": "Vision API సిద్ధం",
        "status_vision_off": "Vision API అందుబాటులో లేదు",
        "tab_text": "📝 వచన విశ్లేషణ",
        "tab_image": "🖼️ చిత్ర విశ్లేషణ",
        "tab_batch": "⚡ బ్యాచ్ విశ్లేషణ",
        "input_label": "మీ ఆలోచనలు లేదా భావాలు నమోదు చేయండి:",
        "input_placeholder": "ఇక్కడ తెలుగులో టైప్ చేయండి...",
        "btn_analyze": "🔍 ఇప్పుడు విశ్లేషించండి",
        "spinner_translate": "ఆఫ్‌లైన్‌లో అనువదిస్తున్నారు...",
        "spinner_bert": "BERT విశ్లేషణ జరుగుతోంది...",
        "result_title": "నిర్ధారణ ఫలితం",
        "confidence_title": "విశ్వాస స్థాయి",
        "disclaimer": "⚠️ పరిశోధన నమూనా మాత్రమే. వైద్య సాధనం కాదు.",
        "img_upload": "చిత్రం అప్‌లోడ్ చేయండి (JPG, PNG, WEBP, BMP):",
        "btn_vision": "🔬 చిత్రాన్ని విశ్లేషించండి",
        "spinner_vision": "Groq Vision ద్వారా విశ్లేషిస్తున్నారు...",
        "batch_label": "బహుళ ఎంట్రీలను అతికించండి (ప్రతి పంక్తి ఒకటి):",
        "batch_placeholder": "పంక్తి 1 (తెలుగు)\nLine 2 (English)...",
        "btn_batch": "⚡ బ్యాచ్ విశ్లేషణ అమలు చేయండి",
        "batch_summary": "బ్యాచ్ సారాంశం",
        "total": "మొత్తం రికార్డులు",
        "depression": "నిరాశ ప్రమాదం",
        "anxiety": "ఆందోళన ప్రమాదం",
        "normal": "సాధారణం",
        "download_csv": "⬇️ CSV డౌన్‌లోడ్",
        "download_txt": "⬇️ నివేదిక డౌన్‌లోడ్",
        "advice_normal": "గణనీయమైన ప్రమాదం కనుగొనబడలేదు. ఆరోగ్యకరమైన జీవితాన్ని కొనసాగించండి.",
        "advice_anxiety": "ఆందోళన సంకేతాలు కనుగొనబడ్డాయి. మానసిక ఆరోగ్య నిపుణుడితో మాట్లాడండి.",
        "advice_depression": "నిరాశ సంకేతాలు కనుగొనబడ్డాయి. మానసిక ఆరోగ్య నిపుణుడిని సంప్రదించండి.",
        "choose_lang": "భాషను ఎంచుకోండి",
        "model_info": "Fine-tuned BERT · Reddit Mental Health Dataset",
        "offline_badge": "🔒 ఆఫ్‌లైన్ అనువాదం",
    },
    "ml": {
        "flag": "🇮🇳", "name": "മലയാളം",
        "tagline": "ബഹുഭാഷാ BERT മാനസിക ആരോഗ്യ അപകട വിശകലനം",
        "caption": "മൈസൂർ സർവ്വകലാശാല · AI&DS · Enterprise V2.0",
        "status_bert": "BERT മോഡൽ തയ്യാർ",
        "status_vision": "Vision API തയ്യാർ",
        "status_vision_off": "Vision API ലഭ്യമല്ല",
        "tab_text": "📝 ടെക്സ്റ്റ് വിശകലനം",
        "tab_image": "🖼️ ചിത്ര വിശകലനം",
        "tab_batch": "⚡ ബാച്ച് വിശകലനം",
        "input_label": "നിങ്ങളുടെ ചിന്തകൾ അല്ലെങ്കിൽ വികാരങ്ങൾ നൽകുക:",
        "input_placeholder": "ഇവിടെ മലയാളത്തിൽ ടൈപ്പ് ചെയ്യൂ...",
        "btn_analyze": "🔍 ഇപ്പോൾ വിശകലനം ചെയ്യൂ",
        "spinner_translate": "ഓഫ്‌ലൈൻ പരിഭാഷ...",
        "spinner_bert": "BERT വിശകലനം നടക്കുന്നു...",
        "result_title": "നിർണ്ണയ ഫലം",
        "confidence_title": "വിശ്വസ്തത നില",
        "disclaimer": "⚠️ ഗവേഷണ മാതൃക മാത്രം. വൈദ്യശാസ്ത്ര ഉപകരണമല്ല.",
        "img_upload": "ചിത്രം അപ്‌ലോഡ് ചെയ്യൂ (JPG, PNG, WEBP, BMP):",
        "btn_vision": "🔬 ചിത്രം വിശകലനം ചെയ്യൂ",
        "spinner_vision": "Groq Vision വഴി വിശകലനം...",
        "batch_label": "ഒന്നിലധികം എൻട്രികൾ ഒട്ടിക്കൂ (ഓരോ വരിയും ഒന്ന്):",
        "batch_placeholder": "വരി 1 (മലയാളം)\nLine 2 (English)...",
        "btn_batch": "⚡ ബാച്ച് വിശകലനം പ്രവർത്തിപ്പിക്കൂ",
        "batch_summary": "ബാച്ച് സംഗ്രഹം",
        "total": "ആകെ രേഖകൾ",
        "depression": "വിഷാദ അപകടം",
        "anxiety": "ഉത്കണ്ഠ അപകടം",
        "normal": "സാധാരണ",
        "download_csv": "⬇️ CSV ഡൗൺലോഡ്",
        "download_txt": "⬇️ റിപ്പോർട്ട് ഡൗൺലോഡ്",
        "advice_normal": "കാര്യമായ അപകടം കണ്ടെത്തിയില്ല. ആരോഗ്യകരമായ ജീവിതം തുടരൂ.",
        "advice_anxiety": "ഉത്കണ്ഠ ലക്ഷണങ്ങൾ കണ്ടെത്തി. ഒരു മാനസിക ആരോഗ്യ വിദഗ്ദ്ധനോട് സംസാരിക്കൂ.",
        "advice_depression": "വിഷാദ ലക്ഷണങ്ങൾ കണ്ടെത്തി. ദയവായി ഒരു വിദഗ്ദ്ധനെ സമീപിക്കൂ.",
        "choose_lang": "ഭാഷ തിരഞ്ഞെടുക്കൂ",
        "model_info": "Fine-tuned BERT · Reddit Mental Health Dataset",
        "offline_badge": "🔒 ഓഫ്‌ലൈൻ പരിഭാഷ",
    },
    "bn": {
        "flag": "🇮🇳", "name": "বাংলা",
        "tagline": "বহুভাষিক BERT মানসিক স্বাস্থ্য ঝুঁকি বিশ্লেষণ",
        "caption": "মহীশূর বিশ্ববিদ্যালয় · AI&DS · Enterprise V2.0",
        "status_bert": "BERT মডেল প্রস্তুত",
        "status_vision": "Vision API প্রস্তুত",
        "status_vision_off": "Vision API অনুপলব্ধ",
        "tab_text": "📝 পাঠ্য বিশ্লেষণ",
        "tab_image": "🖼️ ছবি বিশ্লেষণ",
        "tab_batch": "⚡ ব্যাচ বিশ্লেষণ",
        "input_label": "আপনার চিন্তা বা অনুভূতি লিখুন:",
        "input_placeholder": "এখানে বাংলায় টাইপ করুন...",
        "btn_analyze": "🔍 এখন বিশ্লেষণ করুন",
        "spinner_translate": "অফলাইনে অনুবাদ হচ্ছে...",
        "spinner_bert": "BERT বিশ্লেষণ চলছে...",
        "result_title": "রোগ নির্ণয় ফলাফল",
        "confidence_title": "আস্থার মাত্রা",
        "disclaimer": "⚠️ গবেষণা প্রোটোটাইপ শুধু। চিকিৎসা সরঞ্জাম নয়।",
        "img_upload": "ছবি আপলোড করুন (JPG, PNG, WEBP, BMP):",
        "btn_vision": "🔬 ছবি বিশ্লেষণ করুন",
        "spinner_vision": "Groq Vision এর মাধ্যমে বিশ্লেষণ...",
        "batch_label": "একাধিক এন্ট্রি পেস্ট করুন (প্রতি লাইনে একটি):",
        "batch_placeholder": "লাইন ১ (বাংলা)\nLine 2 (English)...",
        "btn_batch": "⚡ ব্যাচ বিশ্লেষণ চালান",
        "batch_summary": "ব্যাচ সারসংক্ষেপ",
        "total": "মোট রেকর্ড",
        "depression": "বিষণ্নতা ঝুঁকি",
        "anxiety": "উদ্বেগ ঝুঁকি",
        "normal": "স্বাভাবিক",
        "download_csv": "⬇️ CSV ডাউনলোড",
        "download_txt": "⬇️ রিপোর্ট ডাউনলোড",
        "advice_normal": "কোনো উল্লেখযোগ্য ঝুঁকি পাওয়া যায়নি। সুস্থ জীবনযাপন অব্যাহত রাখুন।",
        "advice_anxiety": "উদ্বেগের লক্ষণ পাওয়া গেছে। মানসিক স্বাস্থ্য বিশেষজ্ঞের সাথে কথা বলুন।",
        "advice_depression": "বিষণ্নতার লক্ষণ পাওয়া গেছে। দয়া করে একজন বিশেষজ্ঞের সাথে যোগাযোগ করুন।",
        "choose_lang": "ভাষা বেছে নিন",
        "model_info": "Fine-tuned BERT · Reddit Mental Health Dataset",
        "offline_badge": "🔒 অফলাইন অনুবাদ",
    },
    "mr": {
        "flag": "🇮🇳", "name": "मराठी",
        "tagline": "बहुभाषिक BERT मानसिक आरोग्य जोखीम विश्लेषण",
        "caption": "म्हैसूर विद्यापीठ · AI&DS · Enterprise V2.0",
        "status_bert": "BERT मॉडेल तयार",
        "status_vision": "Vision API तयार",
        "status_vision_off": "Vision API अनुपलब्ध",
        "tab_text": "📝 मजकूर विश्लेषण",
        "tab_image": "🖼️ प्रतिमा विश्लेषण",
        "tab_batch": "⚡ बॅच विश्लेषण",
        "input_label": "आपले विचार किंवा भावना नमूद करा:",
        "input_placeholder": "येथे मराठीत टाइप करा...",
        "btn_analyze": "🔍 आता विश्लेषण करा",
        "spinner_translate": "ऑफलाइन भाषांतर होत आहे...",
        "spinner_bert": "BERT विश्लेषण चालू आहे...",
        "result_title": "निदान निकाल",
        "confidence_title": "विश्वास पातळी",
        "disclaimer": "⚠️ फक्त संशोधन प्रोटोटाइप. वैद्यकीय साधन नाही.",
        "img_upload": "प्रतिमा अपलोड करा (JPG, PNG, WEBP, BMP):",
        "btn_vision": "🔬 प्रतिमा विश्लेषण करा",
        "spinner_vision": "Groq Vision द्वारे विश्लेषण...",
        "batch_label": "एकाधिक नोंदी पेस्ट करा (प्रति ओळ एक):",
        "batch_placeholder": "ओळ 1 (मराठी)\nLine 2 (English)...",
        "btn_batch": "⚡ बॅच विश्लेषण चालवा",
        "batch_summary": "बॅच सारांश",
        "total": "एकूण नोंदी",
        "depression": "नैराश्य धोका",
        "anxiety": "चिंता धोका",
        "normal": "सामान्य",
        "download_csv": "⬇️ CSV डाउनलोड",
        "download_txt": "⬇️ अहवाल डाउनलोड",
        "advice_normal": "कोणताही महत्त्वपूर्ण धोका आढळला नाही. निरोगी जीवनशैली सुरू ठेवा.",
        "advice_anxiety": "चिंतेची चिन्हे आढळली. मानसिक आरोग्य तज्ञाशी बोला.",
        "advice_depression": "नैराश्याची चिन्हे आढळली. कृपया तज्ञाशी संपर्क साधा.",
        "choose_lang": "भाषा निवडा",
        "model_info": "Fine-tuned BERT · Reddit Mental Health Dataset",
        "offline_badge": "🔒 ऑफलाइन भाषांतर",
    },
}

# ═══════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════
st.set_page_config(
    page_title="MindScan — AI Mental Health Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ═══════════════════════════════════════════════════
# PREMIUM DYNAMIC CSS
# ═══════════════════════════════════════════════════
PREMIUM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Noto+Sans+Devanagari:wght@400;600;700&family=Noto+Sans+Tamil:wght@400;600;700&family=Noto+Sans+Telugu:wght@400;600;700&family=Noto+Sans+Kannada:wght@400;600;700&family=Noto+Sans+Malayalam:wght@400;600;700&family=Noto+Sans+Bengali:wght@400;600;700&display=swap');

/* ── Global Reset ── */
html, body, [class*="css"], .stMarkdown, p, span, label, input, button, select, textarea {
    font-family: 'Plus Jakarta Sans', 'Outfit', 'Noto Sans Devanagari', 'Noto Sans Tamil',
                 'Noto Sans Telugu', 'Noto Sans Kannada', 'Noto Sans Malayalam',
                 'Noto Sans Bengali', sans-serif !important;
}

/* ── Animated Background ── */
.stApp {
    background: #060810 !important;
    background-image:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(99, 102, 241, 0.18) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(236, 72, 153, 0.12) 0%, transparent 55%),
        radial-gradient(ellipse 50% 50% at 50% 50%, rgba(16, 20, 40, 0.95) 0%, transparent 100%) !important;
}

/* ── Floating particles layer ── */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image:
        radial-gradient(circle 1px at 20% 30%, rgba(139,92,246,0.4) 0%, transparent 2px),
        radial-gradient(circle 1px at 70% 15%, rgba(236,72,153,0.3) 0%, transparent 2px),
        radial-gradient(circle 1px at 45% 70%, rgba(99,102,241,0.35) 0%, transparent 2px),
        radial-gradient(circle 1px at 85% 60%, rgba(139,92,246,0.25) 0%, transparent 2px),
        radial-gradient(circle 1px at 10% 80%, rgba(236,72,153,0.2) 0%, transparent 2px),
        radial-gradient(circle 2px at 60% 40%, rgba(99,102,241,0.2) 0%, transparent 3px),
        radial-gradient(circle 1px at 30% 55%, rgba(168,85,247,0.3) 0%, transparent 2px);
    pointer-events: none;
    z-index: 0;
    animation: drift 20s linear infinite;
}

@keyframes drift {
    0% { transform: translate(0, 0); }
    50% { transform: translate(10px, -15px); }
    100% { transform: translate(0, 0); }
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden !important; }
.block-container { padding-top: 2rem !important; max-width: 1200px; }

/* ── Main Title Hero ── */
.hero-section {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    position: relative;
}

.hero-badge {
    display: inline-block;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 50px;
    padding: 0.35rem 1.2rem;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    color: #a5b4fc;
    text-transform: uppercase;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
}

.hero-title {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 900 !important;
    font-size: 4rem !important;
    line-height: 1.1 !important;
    background: linear-gradient(135deg, #818cf8 0%, #a78bfa 40%, #f472b6 80%, #fb923c 100%);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    letter-spacing: -0.04em !important;
    animation: titleGlow 4s ease-in-out infinite alternate;
}

@keyframes titleGlow {
    0% { filter: brightness(1); }
    100% { filter: brightness(1.2) drop-shadow(0 0 20px rgba(168,85,247,0.4)); }
}

.hero-subtitle {
    font-size: 1.1rem !important;
    color: #9ca3af !important;
    font-weight: 400 !important;
    margin-top: 0.5rem !important;
    letter-spacing: 0.01em;
}

.hero-caption {
    font-size: 0.8rem !important;
    color: #4b5563 !important;
    margin-top: 0.5rem;
    font-weight: 500;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* ── Status Chips ── */
.status-row {
    display: flex;
    gap: 0.75rem;
    justify-content: center;
    flex-wrap: wrap;
    margin: 1.5rem 0;
}

.status-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(15, 17, 30, 0.8);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 50px;
    padding: 0.45rem 1.1rem;
    font-size: 0.82rem;
    font-weight: 600;
    backdrop-filter: blur(12px);
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    transition: all 0.3s ease;
}

.status-chip.online {
    color: #34d399;
    border-color: rgba(52,211,153,0.25);
    box-shadow: 0 0 15px rgba(52,211,153,0.1);
}

.status-chip.offline {
    color: #f87171;
    border-color: rgba(248,113,113,0.25);
}

.pulse-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: currentColor;
    animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.8); }
}

/* ── Language Selector Pills ── */
.lang-section {
    background: rgba(15, 17, 30, 0.6);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 20px;
    padding: 1.5rem;
    margin: 1.5rem 0;
    backdrop-filter: blur(20px);
    box-shadow: 0 8px 40px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.05);
}

.lang-heading {
    text-align: center;
    font-size: 0.75rem;
    font-weight: 700;
    color: #6b7280;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
}

/* ── Tabs ── */
button[data-baseweb="tab"] {
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    color: #6b7280 !important;
    border-bottom: 2px solid transparent !important;
    padding: 0.9rem 1.8rem !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    position: relative;
}

button[data-baseweb="tab"]:hover {
    color: #c4b5fd !important;
    background: rgba(139,92,246,0.08) !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #a78bfa !important;
    border-bottom-color: #a78bfa !important;
    background: rgba(139,92,246,0.1) !important;
}

div[data-testid="stTabs"] > div:first-child {
    border-bottom: 1px solid rgba(255,255,255,0.07) !important;
    gap: 0.25rem !important;
}

/* ── Glass Cards for Analysis sections ── */
.analysis-card {
    background: rgba(13, 15, 28, 0.75);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1rem 0;
    backdrop-filter: blur(20px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.06);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

/* ── CTA Gradient Buttons — exclude file uploader context ── */
div.stButton > button,
div[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #d946ef 100%) !important;
    color: white !important;
    border: none !important;
    padding: 0.8rem 2rem !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.35) !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    letter-spacing: 0.02em !important;
    cursor: pointer !important;
    position: relative !important;
    overflow: hidden !important;
}

div.stButton > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
    transform: translateY(-3px) scale(1.02) !important;
    box-shadow: 0 8px 30px rgba(139, 92, 246, 0.55), 0 0 40px rgba(139,92,246,0.2) !important;
}

div.stButton > button:active,
div[data-testid="stFormSubmitButton"] > button:active {
    transform: translateY(1px) scale(0.98) !important;
}

/* ── File Uploader — COMPLETE RESET (Streamlit 1.57 wraps Upload btn in div.stButton) ── */
div[data-testid="stFileUploader"] div.stButton > button,
div[data-testid="stFileUploader"] div.stButton > button:hover,
div[data-testid="stFileUploader"] div.stButton > button:active,
div[data-testid="stFileUploaderDropzone"] button,
div[data-testid="stFileUploaderDropzone"] button:hover {
    all: unset !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    background: rgba(99,102,241,0.12) !important;
    border: 1px solid rgba(139,92,246,0.4) !important;
    border-radius: 8px !important;
    color: #a5b4fc !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    padding: 0.45rem 1.1rem !important;
    cursor: pointer !important;
    white-space: nowrap !important;
    transition: background 0.2s ease, border-color 0.2s ease !important;
    transform: none !important;
    box-shadow: none !important;
    letter-spacing: 0 !important;
    line-height: 1.5 !important;
    overflow: hidden !important;
    text-overflow: clip !important;
}

div[data-testid="stFileUploader"] div.stButton > button:hover,
div[data-testid="stFileUploaderDropzone"] button:hover {
    background: rgba(99,102,241,0.22) !important;
    border-color: rgba(139,92,246,0.7) !important;
    box-shadow: 0 0 12px rgba(99,102,241,0.2) !important;
}

/* Dropzone area itself */
div[data-testid="stFileUploaderDropzone"],
section[data-testid="stFileUploaderDropzone"] {
    background: rgba(12, 14, 24, 0.75) !important;
    border: 1.5px dashed rgba(139,92,246,0.3) !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
}

div[data-testid="stFileUploaderDropzone"]:hover,
section[data-testid="stFileUploaderDropzone"]:hover {
    border-color: rgba(139,92,246,0.65) !important;
    background: rgba(99,102,241,0.05) !important;
}

/* ── Result Cards ── */
.result-normal {
    background: linear-gradient(135deg, rgba(16,185,129,0.12), rgba(5,150,105,0.08));
    border: 1px solid rgba(16,185,129,0.3);
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 0 30px rgba(16,185,129,0.08);
    animation: resultFade 0.5s ease-out;
}

.result-anxiety {
    background: linear-gradient(135deg, rgba(245,158,11,0.12), rgba(217,119,6,0.08));
    border: 1px solid rgba(245,158,11,0.3);
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 0 30px rgba(245,158,11,0.08);
    animation: resultFade 0.5s ease-out;
}

.result-depression {
    background: linear-gradient(135deg, rgba(239,68,68,0.12), rgba(185,28,28,0.08));
    border: 1px solid rgba(239,68,68,0.3);
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 0 30px rgba(239,68,68,0.08);
    animation: resultFade 0.5s ease-out;
}

@keyframes resultFade {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

.result-label {
    font-family: 'Outfit', sans-serif !important;
    font-size: 1.5rem !important;
    font-weight: 800 !important;
    margin: 0 !important;
    letter-spacing: -0.02em;
}

.result-advice {
    font-size: 0.9rem !important;
    margin-top: 0.4rem !important;
    opacity: 0.85;
}

/* ── Confidence Bars ── */
.conf-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin: 0.5rem 0;
}

.conf-label {
    font-size: 0.85rem;
    font-weight: 600;
    min-width: 140px;
    color: #d1d5db;
}

.conf-bar-bg {
    flex: 1;
    height: 8px;
    background: rgba(255,255,255,0.07);
    border-radius: 10px;
    overflow: hidden;
}

.conf-bar-fill {
    height: 100%;
    border-radius: 10px;
    transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
}

.conf-pct {
    font-size: 0.85rem;
    font-weight: 700;
    min-width: 45px;
    text-align: right;
    color: #9ca3af;
}

/* ── Text Areas ── */
textarea, input[type="text"] {
    background: rgba(12, 14, 24, 0.9) !important;
    color: #e5e7eb !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    transition: border-color 0.3s, box-shadow 0.3s !important;
    font-size: 0.95rem !important;
}

textarea:focus, input[type="text"]:focus {
    border-color: rgba(139,92,246,0.6) !important;
    box-shadow: 0 0 0 3px rgba(139,92,246,0.15) !important;
}

/* ── Selectbox ── */
div[data-testid="stSelectbox"] > div > div {
    background: rgba(12, 14, 24, 0.9) !important;
    border-color: rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    color: #e5e7eb !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #07080e !important;
    border-right: 1px solid rgba(255,255,255,0.04) !important;
}

/* ── Progress bars ── */
div[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #6366f1, #a855f7) !important;
    border-radius: 10px !important;
}

/* ── Metrics ── */
div[data-testid="metric-container"] {
    background: rgba(15,17,30,0.7) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 14px !important;
    padding: 1.2rem !important;
    backdrop-filter: blur(10px) !important;
    transition: transform 0.2s ease !important;
}

div[data-testid="metric-container"]:hover {
    transform: translateY(-2px) !important;
}

div[data-testid="stMetricValue"] {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 800 !important;
    font-size: 1.8rem !important;
}

/* ── Dividers ── */
hr {
    border-color: rgba(255,255,255,0.06) !important;
    margin: 1.5rem 0 !important;
}

/* ── Alerts ── */
div[data-testid="stAlert"] {
    border-radius: 12px !important;
    backdrop-filter: blur(10px) !important;
}

/* ── Dataframe ── */
div[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
}

/* ── Image ── */
img {
    border-radius: 12px !important;
}

/* ── Offline badge ── */
.offline-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(99,102,241,0.12);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 50px;
    padding: 0.2rem 0.8rem;
    font-size: 0.78rem;
    font-weight: 600;
    color: #a5b4fc;
    margin-left: 0.5rem;
}

/* ── Analysis Mode Radio (tab 2 only) ── */
div[data-testid="stRadio"] > div {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 0.6rem !important;
}

div[data-testid="stRadio"] label {
    background: rgba(15,17,30,0.8) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 50px !important;
    padding: 0.45rem 1.2rem !important;
    cursor: pointer !important;
    transition: all 0.25s ease !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    color: #9ca3af !important;
    white-space: nowrap !important;
}

div[data-testid="stRadio"] label:hover {
    background: rgba(139,92,246,0.15) !important;
    border-color: rgba(139,92,246,0.4) !important;
    color: #c4b5fd !important;
}
}
</style>
"""

st.markdown(PREMIUM_CSS, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════
# LANGUAGE SELECTION (SESSION STATE — persists UI state)
# ═══════════════════════════════════════════════════
if "ui_lang" not in st.session_state:
    st.session_state.ui_lang = "en"


# ═══════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════
MODEL_DIR = Config.MODEL_DIR
DEVICE    = torch.device(Config.DEVICE)
LABELS    = ["anxiety", "depression", "normal"]

# ═══════════════════════════════════════════════════
# CACHED RESOURCES
# ═══════════════════════════════════════════════════
@st.cache_resource
def load_bert():
    if not os.path.exists(MODEL_DIR):
        logger.error(f"BERT model directory not found at {MODEL_DIR}")
        return None, None
    try:
        tok   = BertTokenizer.from_pretrained(MODEL_DIR)
        model = BertForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)
        model.eval()
        return tok, model
    except Exception as e:
        logger.exception("Failed to load BERT model.")
        return None, None

@st.cache_resource
def get_translation_service():
    return TranslationService()

@st.cache_resource
def get_vision_service():
    try:
        return GroqVisionService()
    except Exception as e:
        logger.warning(f"GroqVisionService init failed: {e}")
        return None

def predict_text(text, tok, model):
    enc = tok(text, max_length=128, padding="max_length",
              truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(
            input_ids      = enc["input_ids"].to(DEVICE),
            attention_mask = enc["attention_mask"].to(DEVICE),
        ).logits
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return LABELS[int(np.argmax(probs))], probs

# ═══════════════════════════════════════════════════
# LOAD SERVICES
# ═══════════════════════════════════════════════════
tok, bert_model     = load_bert()
translator_svc      = get_translation_service()
vision_svc          = get_vision_service()

# ═══════════════════════════════════════════════════
# HERO SECTION
# ═══════════════════════════════════════════════════
# Resolve current UI strings
lang = st.session_state.ui_lang
S = UI_STRINGS.get(lang, UI_STRINGS["en"])

st.markdown(f"""
<div class="hero-section">
    <div class="hero-badge">🧠 AI-Powered Mental Health Insights</div>
    <h1 class="hero-title">MindScan</h1>
    <p class="hero-subtitle">{S['tagline']}</p>
    <p class="hero-caption">{S['caption']}</p>
</div>
""", unsafe_allow_html=True)

# ── Status Chips ──
bert_ok    = bert_model is not None
vision_ok  = vision_svc is not None

bert_chip  = f"""<div class="status-chip {'online' if bert_ok else 'offline'}">
    <span class="pulse-dot"></span> 🤖 {S['status_bert'] if bert_ok else 'BERT Unavailable'}
</div>"""
vision_chip = f"""<div class="status-chip {'online' if vision_ok else 'offline'}">
    <span class="pulse-dot"></span> ⚡ {S['status_vision'] if vision_ok else S['status_vision_off']}
</div>"""
device_chip = f"""<div class="status-chip online">
    <span class="pulse-dot"></span> 💻 {str(DEVICE).upper()}
</div>"""

st.markdown(f'<div class="status-row">{bert_chip}{vision_chip}{device_chip}</div>', unsafe_allow_html=True)

if not bert_ok:
    st.error(f"BERT model not found at: `{MODEL_DIR}` — Please verify model weights.")
    st.stop()

# ═══════════════════════════════════════════════════
# LANGUAGE SELECTOR (2-row grid — works on all scripts)
# ═══════════════════════════════════════════════════
LANG_OPTIONS = {
    "en": {"label": "English",   "emoji": "EN", "native": "English"},
    "hi": {"label": "Hindi",     "emoji": "HI", "native": "हिन्दी"},
    "kn": {"label": "Kannada",   "emoji": "KN", "native": "ಕನ್ನಡ"},
    "ta": {"label": "Tamil",     "emoji": "TA", "native": "தமிழ்"},
    "te": {"label": "Telugu",    "emoji": "TE", "native": "తెలుగు"},
    "ml": {"label": "Malayalam", "emoji": "ML", "native": "മലയാളം"},
    "bn": {"label": "Bengali",   "emoji": "BN", "native": "বাংলা"},
    "mr": {"label": "Marathi",   "emoji": "MR", "native": "मराठी"},
}

st.markdown(f'<p style="text-align:center;color:#6b7280;font-size:0.72rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.6rem;">🌐 {S["choose_lang"]}</p>', unsafe_allow_html=True)

lang_codes = list(LANG_OPTIONS.keys())
# Two rows of 4 so buttons are wider and text doesn't wrap
row1 = lang_codes[:4]   # en, hi, kn, ta
row2 = lang_codes[4:]   # te, ml, bn, mr

for row in [row1, row2]:
    cols = st.columns(4)
    for idx, code in enumerate(row):
        info = LANG_OPTIONS[code]
        is_active = (code == st.session_state.ui_lang)
        # Clean label: tick + native name for active, just native name for inactive
        btn_label = f"✓  {info['native']}" if is_active else info['native']
        if cols[idx].button(
            btn_label,
            key=f"lang_btn_{code}",
            use_container_width=True,
            type="primary" if is_active else "secondary"
        ):
            if code != st.session_state.ui_lang:
                st.session_state.ui_lang = code
                st.rerun()

# Refresh S after potential lang change
lang = st.session_state.ui_lang
S = UI_STRINGS.get(lang, UI_STRINGS["en"])
is_multilingual = (lang != "en")

# ── Model Architecture Note ──
if is_multilingual:
    st.markdown(f"""
<div style="background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.2);border-radius:12px;
            padding:0.8rem 1.1rem;margin:0.8rem 0;display:flex;align-items:center;gap:0.8rem;">
    <span style="font-size:1.4rem;">🔄</span>
    <div>
        <p style="color:#a5b4fc;font-weight:700;font-size:0.85rem;margin:0;">Translation Pipeline Active</p>
        <p style="color:#6b7280;font-size:0.78rem;margin:0;">
            Your {LANG_OPTIONS[lang]['label']} input will be translated to English offline (Helsinki-NLP)
            → then analyzed by BERT (trained on English mental health data). 100% private — no cloud.
        </p>
    </div>
</div>""", unsafe_allow_html=True)

st.divider()

# ═══════════════════════════════════════════════════
# MAIN TABS
# ═══════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([S["tab_text"], S["tab_image"], S["tab_batch"]])

# ─────────────────────────────────────────
# TAB 1 — TEXT ANALYSIS
# ─────────────────────────────────────────
with tab1:
    st.markdown(f"### {S['tab_text']}")
    st.caption(S["model_info"])

    text_input = st.text_area(
        S["input_label"],
        height=150,
        placeholder=S["input_placeholder"],
        key="text_analysis_input"
    )

    analyze_clicked = st.button(S["btn_analyze"], type="primary", use_container_width=True, key="btn_analyze_text")

    if analyze_clicked:
        if not text_input.strip():
            st.warning("⚠️ Please enter some text to analyze." if lang == "en" else S["input_label"])
        else:
            text_to_analyze = text_input
            translated_shown = False

            if is_multilingual:
                with st.spinner(S["spinner_translate"]):
                    try:
                        translated_text = translator_svc.translate(text_input)
                        if translated_text:
                            st.markdown(
                                f'<div style="background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.3);'
                                f'border-radius:10px;padding:0.8rem 1rem;margin:0.5rem 0;font-size:0.9rem;">'
                                f'<span class="offline-badge">{S["offline_badge"]}</span> '
                                f'<strong style="color:#e5e7eb;">{translated_text}</strong></div>',
                                unsafe_allow_html=True
                            )
                            text_to_analyze = translated_text
                            translated_shown = True
                    except Exception as e:
                        st.error(f"Translation failed: {e}")

            with st.spinner(S["spinner_bert"]):
                label_pred, probs = predict_text(text_to_analyze, tok, bert_model)

            st.divider()
            st.markdown(f"#### {S['result_title']}")

            advice_map = {
                "normal":     S["advice_normal"],
                "anxiety":    S["advice_anxiety"],
                "depression": S["advice_depression"],
            }
            icons_map = {
                "normal":     ("✅", "NORMAL",          "result-normal",     "#34d399"),
                "anxiety":    ("⚠️", "ANXIETY RISK",    "result-anxiety",    "#fbbf24"),
                "depression": ("🔴", "DEPRESSION RISK", "result-depression", "#f87171"),
            }
            icon, label_text, css_class, color = icons_map[label_pred]

            st.markdown(f"""
<div class="{css_class}">
    <p class="result-label" style="color:{color};">{icon} {label_text}</p>
    <p class="result-advice" style="color:{color}80;">{advice_map[label_pred]}</p>
</div>""", unsafe_allow_html=True)

            # Confidence Breakdown
            st.markdown(f"#### {S['confidence_title']}")
            conf_icons = {
                "anxiety":    ("⚠️", "#fbbf24", "linear-gradient(90deg, #f59e0b, #d97706)"),
                "depression": ("🔴", "#f87171", "linear-gradient(90deg, #ef4444, #b91c1c)"),
                "normal":     ("✅", "#34d399", "linear-gradient(90deg, #10b981, #059669)"),
            }
            conf_html = ""
            for lbl, prob in zip(LABELS, probs):
                pct    = float(prob) * 100
                ci, cc, cg = conf_icons[lbl]
                conf_html += f"""
<div class="conf-row">
    <span class="conf-label">{ci} {lbl.capitalize()}</span>
    <div class="conf-bar-bg">
        <div class="conf-bar-fill" style="width:{pct:.1f}%;background:{cg};"></div>
    </div>
    <span class="conf-pct" style="color:{cc};">{pct:.1f}%</span>
</div>"""
            st.markdown(conf_html, unsafe_allow_html=True)

            st.markdown(f'<p style="color:#4b5563;font-size:0.78rem;margin-top:1rem;">{S["disclaimer"]}</p>', unsafe_allow_html=True)

# ─────────────────────────────────────────
# TAB 2 — IMAGE ANALYSIS
# ─────────────────────────────────────────
with tab2:
    st.markdown(f"### {S['tab_image']}")
    st.caption("Powered by Groq Vision — Meta Llama Scout 17B")

    if not vision_svc:
        st.error("Groq Vision Service is unavailable. Check your GROQ_API_KEY in the .env file.")
    else:
        mode_opts = {
            "general":      "🌐 " + ("General Visual Tone" if lang == "en" else "General Analysis"),
            "social_media": "📱 " + ("Social Media Sentiment" if lang == "en" else "Social Media"),
            "chart":        "📊 " + ("Scientific Chart Analysis" if lang == "en" else "Chart / Graph"),
        }
        mode = st.radio(
            "Analysis Mode:",
            options=list(mode_opts.keys()),
            format_func=lambda x: mode_opts[x],
            horizontal=True,
            key="img_analysis_mode"
        )

        mode_prompts = {
            "general": (
                "Analyze this image thoroughly and provide:\n"
                "1. **Scene Description** - Key visual elements, setting, objects, colors.\n"
                "2. **Emotional Tone** - Mood and atmosphere the image conveys.\n"
                "3. **Mental Health Relevance** - Visual cues related to emotional wellbeing. Be empathetic.\n"
                "4. **Key Insights** - Most important takeaways.\n"
                "Use clear headings. Be thorough and empathetic."
            ),
            "social_media": (
                "Analyze this as a social media post and provide:\n"
                "1. **Visual Content** - People, environment, objects, filters, aesthetic.\n"
                "2. **Emotional Signals** - Emotions via expressions, body language, colors.\n"
                "3. **Mental Health Indicators** - Signs of distress, isolation, or positivity. Rate: Positive / Neutral / Concerning.\n"
                "4. **Context** - What the poster may be communicating.\n"
                "Be empathetic and non-judgmental."
            ),
            "chart": (
                "Analyze this chart or graph and provide:\n"
                "1. **Chart Type and Structure** - Visualization type, axes, scales.\n"
                "2. **Key Data Patterns** - Trends, peaks, dips, anomalies.\n"
                "3. **Mental Health Context** - Interpret mental health metrics if applicable.\n"
                "4. **Conclusions** - Precise summaries and next steps.\n"
                "Be precise and technical."
            ),
        }

        uploaded = st.file_uploader(
            S["img_upload"],
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            key="img_uploader"
        )

        if uploaded:
            try:
                image = Image.open(uploaded)
                col_a, col_b = st.columns([1.2, 1])
                with col_a:
                    st.image(image, caption=uploaded.name, use_column_width=True)
                with col_b:
                    st.markdown(f"""
<div style="background:rgba(15,17,30,0.7);border:1px solid rgba(255,255,255,0.07);border-radius:12px;padding:1.2rem;">
    <p style="color:#6b7280;font-size:0.78rem;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.8rem;">📋 File Metadata</p>
    <p style="color:#d1d5db;margin:0.3rem 0;"><strong>Name:</strong> {uploaded.name}</p>
    <p style="color:#d1d5db;margin:0.3rem 0;"><strong>Resolution:</strong> {image.size[0]} × {image.size[1]} px</p>
    <p style="color:#d1d5db;margin:0.3rem 0;"><strong>Mode:</strong> {image.mode}</p>
    <p style="color:#d1d5db;margin:0.3rem 0;"><strong>Size:</strong> {uploaded.size / 1024:.1f} KB</p>
</div>""", unsafe_allow_html=True)

                if st.button(S["btn_vision"], type="primary", use_container_width=True, key="btn_vision_analyze"):
                    with st.spinner(S["spinner_vision"]):
                        try:
                            result = vision_svc.analyze_image(image, mode_prompts[mode])
                            st.divider()
                            st.markdown("### 📋 Analysis Report")
                            st.markdown(result)
                            st.download_button(
                                label=S["download_txt"],
                                data=result,
                                file_name=f"mindscan_report_{uploaded.name}.txt",
                                mime="text/plain"
                            )
                        except Exception as e:
                            st.error(f"Image analysis failed: {e}")
            except Exception as e:
                st.error(f"Failed to load image: {e}")

# ─────────────────────────────────────────
# TAB 3 — BATCH ANALYSIS
# ─────────────────────────────────────────
with tab3:
    st.markdown(f"### {S['tab_batch']}")
    st.caption("Auto-detects Indian scripts via Unicode range (U+0900–U+0D7F). Translates offline.")

    batch_text = st.text_area(
        S["batch_label"],
        height=200,
        placeholder=S["batch_placeholder"],
        key="batch_text_input"
    )

    if st.button(S["btn_batch"], type="primary", use_container_width=True, key="btn_batch_run"):
        lines = [l.strip() for l in batch_text.strip().split("\n") if len(l.strip()) > 5]
        if not lines:
            st.warning("No valid entries found (minimum 5 characters per line).")
        else:
            results = []
            bar     = st.progress(0, text="Processing batch records...")

            for i, line in enumerate(lines):
                has_indian = any('\u0900' <= c <= '\u0D7F' for c in line)
                analyzed   = line
                lang_tag   = "🇬🇧 English"

                if has_indian:
                    lang_tag = "🇮🇳 Indian Script"
                    try:
                        t = translator_svc.translate(line)
                        if t:
                            analyzed = t
                    except Exception as ex:
                        logger.error(f"Batch translation failed: {ex}")

                lbl, prb = predict_text(analyzed, tok, bert_model)
                results.append({
                    "#"              : i + 1,
                    "Script"         : lang_tag,
                    "Input Text"     : line[:65] + "..." if len(line) > 65 else line,
                    "Prediction"     : lbl.upper(),
                    "Confidence"     : f"{float(prb.max()) * 100:.1f}%",
                    "Risk Level"     : "🔴 High" if lbl == "depression" else
                                       "🟡 Medium" if lbl == "anxiety" else "🟢 Low",
                })
                bar.progress((i + 1) / len(lines), text=f"Record {i+1} of {len(lines)}...")

            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.divider()

            st.markdown(f"#### {S['batch_summary']}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(S["total"],      len(results))
            c2.metric(f"🔴 {S['depression']}", sum(1 for r in results if "DEPRESSION" in r["Prediction"]))
            c3.metric(f"🟡 {S['anxiety']}",    sum(1 for r in results if "ANXIETY"    in r["Prediction"]))
            c4.metric(f"🟢 {S['normal']}",     sum(1 for r in results if "NORMAL"     in r["Prediction"]))

            st.download_button(
                label=S["download_csv"],
                data=df.to_csv(index=False),
                file_name="mindscan_batch_report.csv",
                mime="text/csv"
            )

# ═══════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════
st.divider()
st.markdown("""
<div style="text-align:center;padding:1rem 0 2rem;">
    <p style="color:#374151;font-size:0.78rem;letter-spacing:0.05em;">
        Built with Fine-tuned BERT · Groq Llama 4 Scout · HuggingFace Offline Translation · Streamlit
        <br>Mysore University School of Engineering · AI&DS Department · Phase-I Architecture
    </p>
</div>
""", unsafe_allow_html=True)
