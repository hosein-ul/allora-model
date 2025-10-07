راهنمایی کامل از صفر برای ران کردن Worker در Allora Network روی سرور خالی (VPS)سلام! اول عذرخواهی می‌کنم اگر توضیحات قبلی ناقص بود – حق داری، من باید از اول کامل بگم. تو هیچی نمی‌دونی، پس فرض می‌کنم سرورت (مثل VPS از Kamatera یا Oracle) کاملاً خالیه (Ubuntu 22.04 LTS نصب‌شده، بدون هیچی). من همه چیز رو از صفر توضیح می‌دم: از SSH وصل شدن تا ران کردن worker برای topic 69 (پیش‌بینی ۲۴ ساعته BTC). نکته کلیدی در مورد والت (کیف پول): والت برای شرکت در شبکه Allora لازمه (تا workerت شناسایی بشه و rewards بگیره). در کلاینت worker، اگر mnemonic (عبارتی ۱۲-۲۴ کلمه‌ای) ندی، خودکار generate می‌شه. mnemonic و آدرس والت در فایل .allora_key (در home directory) ذخیره می‌شه. حتماً این فایل رو backup کن (مثل کپی به جای امن) – اگر گم بشه، والتت رو از دست می‌دی. آدرس والت (public address) رو می‌تونی در explorer.allora.network/topics/69 سرچ کنی تا workerت رو ببینی. mnemonic رو فقط خودت بدون، هرگز share نکن.پیش‌نیازها قبل از شروع:API key: از Discord (لینک: https://discord.com/invite/allora) بگیر – در #dev-support پیام بده: "Need API key for Forge Builder Kit and topic 69 worker". رایگان و سریع (چند ساعت).

سرور: Ubuntu 22.04، حداقل ۲GB RAM، ۱ vCPU (از trialهایی مثل Oracle free tier). SSH access داشته باش (root یا user با sudo).

ابزارها: یک ترمینال (مثل PuTTY یا terminal محلی) برای SSH.



حالا بریم گام‌به‌گام. همه دستورات رو کپی-پیست کن.گام ۱: وصل شدن به سرور (SSH) و بروزرسانی اولیه (۵ دقیقه)از لپ‌تاپت، SSH بزن: ssh username@your_server_ip (username معمولاً ubuntu یا root، IP از پنل VPS بگیر، password یا key رو وارد کن).

بروزرسانی سیستم:



sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv git curl wget


کاربر معمولی بساز (اگر root هستی): adduser yourusername و usermod -aG sudo yourusername. بعد su - yourusername بزن.



گام ۲: setup محیط پایتون و clone repo (۱۰ دقیقه)venv بساز (برای ایزوله کردن):



cd ~
python3 -m venv allora_env
source allora_env/bin/activate


pip بروز کن: pip install --upgrade pip.

repo رو clone کن:



git clone https://github.com/allora-network/allora-forge-builder-kit.git
cd allora-forge-builder-kit
pip install -y -r requirements.txt  # اگر فایل requirements باشه، وگرنه دستی: pip install lightgbm pandas dill asyncio scikit-learn
pip install allora_forge_builder_kit  # پکیج اصلی


گام ۳: اسکریپت bash برای setup خودکار (اختیاری، اما راحت – ۲ دقیقه)این اسکریپت همه گام‌های ۱ و ۲ رو automate می‌کنه. در home directory (cd ~) یک فایل بساز: nano setup_allora.sh، کد زیر رو کپی کن، save کن (Ctrl+O, Enter, Ctrl+X)، و ران کن: bash setup_allora.sh.bash



#!/bin/bash

# بروزرسانی سیستم
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv git curl wget

# venv و pip
cd ~
python3 -m venv allora_env
source allora_env/bin/activate
pip install --upgrade pip

# clone و install
git clone https://github.com/allora-network/allora-forge-builder-kit.git
cd allora-forge-builder-kit
pip install lightgbm pandas dill asyncio scikit-learn
pip install allora_forge_builder_kit  # اگر error داد، از source install: pip install -e .

echo "Setup کامل! حالا source ~/allora_env/bin/activate بزن و به گام بعدی برو."



گام ۴: اسکریپت پایتون کامل برای train مدل، export، و دیپلوی worker (۲۰-۳۰ دقیقه اول، بعد مداوم)حالا یک اسکریپت تک‌فایل پایتون می‌نویسم که همه کارها رو انجام می‌ده: fetch داده، feature engineering قوی (بر اساس داکیومنت)، train مدل با grid search (قوی‌تر از پایه)، evaluate، export predict_fn، generate والت (اگر نباشه)، و ران worker async برای topic 69. در دایرکتوری repo (cd ~/allora-forge-builder-kit)، فایل بساز: nano deploy_worker.py، کد زیر رو کپی کن، save کن. مهم: YOUR_API_KEY_HERE رو با key واقعی جایگزین کن. اگر mnemonic داری، در wallet_mnemonic بگذار، وگرنه خالی بذار تا auto-generate بشه.python



#!/usr/bin/env python3
# اسکریپت کامل برای train و دیپلوی worker در topic 69
# ران: python3 deploy_worker.py

from allora_forge_builder_kit import AlloraMLWorkflow, get_api_key, AlloraWorker
import lightgbm as lgb
import pandas as pd
import numpy as np
import dill
import asyncio
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# تنظیمات (API key رو جایگزین کن)
API_KEY = "YOUR_API_KEY_HERE"  # از Discord بگیر
tickers = ["btcusd"]
hours_needed = 24 * 14  # lookback قوی
number_of_input_candles = 24 * 14
target_length = 24  # topic 69

# گام ۱: Workflow
workflow = AlloraMLWorkflow(
    data_api_key=API_KEY,
    tickers=tickers,
    hours_needed=hours_needed,
    number_of_input_candles=number_of_input_candles,
    target_length=target_length
)

# گام ۲: Fetch داده‌ها
X_train, y_train, X_val, y_val, X_test, y_test = workflow.get_train_validation_test_data(
    from_month="2023-01",
    validation_months=6,
    test_months=6
)

# گام ۳: Advanced Features (قوی‌تر)
def advanced_features(df):
    df['close_ma_7'] = df['close'].rolling(7).mean()
    df['close_std_7'] = df['close'].rolling(7).std()
    df['volume_ma_7'] = df['volume'].rolling(7).mean()
    for lag in [1, 7]:
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
        df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
    df['volatility'] = (df['high'] - df['low']) / df['close']
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    return df.dropna()

# اعمال features
train_df = advanced_features(pd.concat([X_train.assign(target=y_train), pd.DataFrame()], axis=1))
val_df = advanced_features(pd.concat([X_val.assign(target=y_val), pd.DataFrame()], axis=1))
test_df = advanced_features(pd.concat([X_test.assign(target=y_test), pd.DataFrame()], axis=1))

feature_cols = [col for col in train_df.columns if col not in ['date', 'target', 'open', 'high', 'low', 'close', 'volume']]
X_train_adv = train_df[feature_cols]
y_train_adv = train_df['target']
X_val_adv = val_df[feature_cols]
y_val_adv = val_df['target']
X_test_adv = test_df[feature_cols]
y_test_adv = test_df['target']

# Normalize
scaler = StandardScaler()
X_train_adv_scaled = pd.DataFrame(scaler.fit_transform(X_train_adv), columns=feature_cols, index=X_train_adv.index)
X_val_adv_scaled = pd.DataFrame(scaler.transform(X_val_adv), columns=feature_cols, index=X_val_adv.index)
X_test_adv_scaled = pd.DataFrame(scaler.transform(X_test_adv), columns=feature_cols, index=X_test_adv.index)

print("داده‌ها آماده شد!")

# گام ۴: Grid Search و Train (قوی‌تر)
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.05],
    'max_depth': [5, 7],
    'num_leaves': [16, 32]
}
base_model = lgb.LGBMRegressor(objective='regression', random_state=42)
grid_search = GridSearchCV(base_model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=1)  # n_jobs=1 برای سرور ضعیف
grid_search.fit(pd.concat([X_train_adv_scaled, X_val_adv_scaled]), pd.concat([y_train_adv, y_val_adv]))

best_model = grid_search.best_estimator_
print(f"بهترین پارامترها: {grid_search.best_params_}")

# گام ۵: Evaluate
test_preds = best_model.predict(X_test_adv_scaled)
rmse = np.sqrt(mean_squared_error(y_test_adv, test_preds))
correlation = np.corrcoef(y_test_adv, test_preds)[0, 1]
directional_acc = ((test_preds > 0) == (y_test_adv > 0)).mean()
print(f"RMSE: {rmse:.4f}, Correlation: {correlation:.4f}, Directional Acc: {directional_acc:.2%}")

metrics = workflow.evaluate_test_data(pd.Series(test_preds, index=X_test_adv_scaled.index))
print("Metrics:", metrics)

# گام ۶: Predict Function و Export
def strong_predict():
    live_df = workflow.get_live_features("btcusd")
    live_df = advanced_features(live_df)
    live_features = live_df[feature_cols]
    live_scaled = scaler.transform(live_features)
    live_preds = best_model.predict(live_scaled)
    return pd.Series(live_preds, index=live_features.index)

# ذخیره مدل
model_data = {'model': best_model, 'scaler': scaler, 'feature_cols': feature_cols, 'advanced_features': advanced_features}
with open("strong_model.pkl", "wb") as f:
    dill.dump(model_data, f)
print("مدل ذخیره شد!")

# گام ۷: دیپلوی Worker (async، topic 69)
async def run_worker():
    with open("strong_model.pkl", "rb") as f:
        data = dill.load(f)
    
    def predict_fn():
        return strong_predict()
    
    worker = AlloraWorker(
        topic_id=69,  # topic 69
        predict_fn=predict_fn,
        api_key=API_KEY,
        wallet_mnemonic=None  # خالی = auto-generate والت
    )
    print("Worker شروع شد! والت generate شد (فایل .allora_key رو چک کن).")
    async for result in worker.run():
        if isinstance(result, Exception):
            print(f"خطا: {result}")
        else:
            print(f"پیش‌بینی submit شد: {result.prediction} در epoch {result.epoch}")

# ران worker (مداوم)
if __name__ == "__main__":
    asyncio.run(run_worker())



چطور ران کن: source ~/allora_env/bin/activate (هر بار)، بعد cd ~/allora-forge-builder-kit و python3 deploy_worker.py. اول train می‌کنه (۱۰-۲۰ دقیقه)، بعد worker مداوم ران می‌شه (پیش‌بینی هر epoch submit می‌کنه).

والت چک کن: بعد از ران، cat ~/.allora_key بزن – mnemonic و آدرس رو می‌بینی. backup: cp ~/.allora_key ~/backup_wallet.txt.



گام ۵: نگه داشتن worker ۲۴/۷ (۵ دقیقه)برای مداوم: screen -S worker (نصب اگر نداری: sudo apt install screen)، بعد اسکریپت رو ران کن، Ctrl+A+D برای detach. برگرد: screen -r worker.

یا systemd service بساز (پیشرفته): فایل /etc/systemd/system/allora-worker.service با محتوای:



[Unit]
Description=Allora Worker
After=network.target

[Service]
User=yourusername
WorkingDirectory=/home/yourusername/allora-forge-builder-kit
ExecStart=/home/yourusername/allora_env/bin/python3 deploy_worker.py
Restart=always

[Install]
WantedBy=multi-user.target



بعد sudo systemctl daemon-reload && sudo systemctl start allora-worker && sudo systemctl enable allora-worker.



troubleshootingارور API key: key رو چک کن یا دوباره از Discord بگیر.

ارور والت: اگر generate نشد، mnemonic دستی بساز (با bip39 tool) و بگذار.

RAM کم: grid search رو n_jobs=1 نگه دار.

لاگ‌ها: در کنسول می‌بینی، یا tail -f /var/log/syslog.



حالا کامل شد – از صفر ران کن، اگر ارور دادی، خروجی رو بفرست. موفق باشی! 

