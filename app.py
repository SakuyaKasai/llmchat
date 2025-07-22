import streamlit as st
import anthropic
import google.generativeai as genai
import openai
import json
import uuid
from datetime import datetime
import time
import os

# ページ設定
st.set_page_config(
    page_title="おはなし",
    page_icon="🤖",
    layout="wide"
)

# CSS スタイリング
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 20px 0;
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
        border-radius: 10px;
        margin-bottom: 30px;
    }
    
    .claude-msg {
        background: #ffe4cc;
        color: #8b4513;
        padding: 8px 12px;
        border-radius: 8px;
        margin: 5px 0;
        margin-right: 20%;
        border-left: 3px solid #ff8c42;
    }
    
    .gemini-msg {
        background: #e6f3ff;
        color: #2c5aa0;
        padding: 8px 12px;
        border-radius: 8px;
        margin: 5px 0;
        margin-right: 20%;
        border-left: 3px solid #4285f4;
    }
    
    .gpt-msg {
        background: #f5f5f5;
        color: #424242;
        padding: 8px 12px;
        border-radius: 8px;
        margin: 5px 0;
        margin-right: 20%;
        border-left: 3px solid #757575;
    }
    
    .topic-display {
        background: linear-gradient(45deg, #4ecdc4, #44a08d);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin: 15px 0;
        font-weight: bold;
        font-size: 16px;
    }
    
    .stats-box {
        background: #f8f9fa;
        padding: 10px;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
        font-size: 14px;
    }
    
    .topic-btn {
        background: none;
        border: 1px solid #ddd;
        padding: 5px 10px;
        margin: 2px;
        border-radius: 15px;
        cursor: pointer;
        font-size: 12px;
        color: #666;
    }
    
    .topic-btn:hover {
        background: #f0f0f0;
        border-color: #667eea;
    }
</style>
""", unsafe_allow_html=True)

# 定数定義
MAX_ROUNDS = 10
CHAR_LIMIT = 200
MAX_TOKENS_OUTPUT = 120
CONTEXT_MESSAGES = 3

# おすすめ話題集
SUGGESTED_TOPICS = {
    "🤖 技術・AI": [
        "プログラミング初心者への教え方",
        "AIが人間を置き換える分野と置き換えない分野",
        "コードレビューで一番大事なこと",
        "リモートワーク vs オフィスワーク"
    ],
    "🧠 哲学・抽象": [
        "創造性とは何か",
        "効率と人間らしさのバランス",
        "完璧なシステムは良いシステムか",
        "データと直感、どちらを信じるべきか"
    ],
    "📅 日常・実用": [
        "理想的な朝のルーティン",
        "時間管理の最強メソッド",
        "在宅勤務の集中環境作り",
        "学習効率を上げる方法"
    ],
    "⚡ 対立・議論": [
        "完璧主義 vs 完了主義",
        "計画派 vs 臨機応変派",
        "シンプル vs 高機能",
        "個人最適化 vs チーム最適化"
    ]
}

# セッション状態の初期化
def init_session_state():
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'current_round' not in st.session_state:
        st.session_state.current_round = 0
    if 'current_topic' not in st.session_state:
        st.session_state.current_topic = ""
    if 'conversation_active' not in st.session_state:
        st.session_state.conversation_active = False
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]
    if 'topic_input' not in st.session_state:
        st.session_state.topic_input = ""

def setup_apis():
    """APIクライアントのセットアップ"""
    try:
        # Anthropic Claude
        claude_client = None
        if hasattr(st.secrets, 'ANTHROPIC_API_KEY'):
            claude_client = anthropic.Anthropic(api_key=st.secrets['ANTHROPIC_API_KEY'])
        elif 'ANTHROPIC_API_KEY' in os.environ:
            claude_client = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
        
        # Google Gemini
        gemini_client = None
        if hasattr(st.secrets, 'GOOGLE_API_KEY'):
            genai.configure(api_key=st.secrets['GOOGLE_API_KEY'])
            gemini_client = genai.GenerativeModel('gemini-pro')
        elif 'GOOGLE_API_KEY' in os.environ:
            genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
            gemini_client = genai.GenerativeModel('gemini-pro')
        
        # OpenAI GPT
        openai_client = None
        if hasattr(st.secrets, 'OPENAI_API_KEY'):
            openai_client = openai.OpenAI(api_key=st.secrets['OPENAI_API_KEY'])
        elif 'OPENAI_API_KEY' in os.environ:
            openai_client = openai.OpenAI(api_key=os.environ['OPENAI_API_KEY'])
        
        return claude_client, gemini_client, openai_client
    except Exception as e:
        st.error(f"API設定エラー: {str(e)}")
        return None, None, None

def get_claude_response(client, topic, history):
    """Claude からの応答を取得"""
    if not client:
        return "🔥 Claude: APIキーが設定されていません。サンプル応答を表示します。技術的な観点から分析してみると、興味深い課題がありますね。"
    
    try:
        # 最近の会話履歴を取得
        recent_history = history[-CONTEXT_MESSAGES:] if history else []
        context = "\n".join([f"{msg['speaker']}: {msg['content']}" for msg in recent_history])
        
        prompt = f"""話題: {topic}

これまでの会話:
{context}

あなたはClaudeです。以下の条件で応答してください：
- {CHAR_LIMIT}文字以内厳守
- 簡潔で要点を絞った内容"""
        
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=MAX_TOKENS_OUTPUT,
            messages=[{"role": "user", "content": prompt}]
        )
        
        content = response.content[0].text.strip()
        if len(content) > CHAR_LIMIT:
            content = content[:CHAR_LIMIT-3] + "..."
        
        return content
        
    except anthropic.RateLimitError:
        st.error("🚫 Claude の利用制限に達しました。しばらくお待ちください。")
        return None
    except Exception as e:
        if "quota" in str(e).lower() or "billing" in str(e).lower():
            st.balloons()
            st.error("🎉 Claude のトークンが不足しました。おしまいです！")
            return None
        return f"🔥 Claude: エラーが発生しました。サンプル応答: {topic}について深く考察してみると、文脈的な観点から興味深い洞察が得られますね。"

def get_gemini_response(client, topic, history):
    """Gemini からの応答を取得"""
    if not client:
        return "💎 Gemini: APIキーが設定されていません。サンプル応答を表示します。データ分析の結果、効率的なアプローチが必要です。"
    
    try:
        recent_history = history[-CONTEXT_MESSAGES:] if history else []
        context = "\n".join([f"{msg['speaker']}: {msg['content']}" for msg in recent_history])
        
        prompt = f"""話題: {topic}

これまでの会話:
{context}

あなたはGeminiです。以下の条件で応答してください：
- {CHAR_LIMIT}文字以内厳守
- 簡潔で的確な内容"""
        
        response = client.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=MAX_TOKENS_OUTPUT,
                temperature=0.7,
            )
        )
        
        if response.text:
            content = response.text.strip()
            if len(content) > CHAR_LIMIT:
                content = content[:CHAR_LIMIT-3] + "..."
            return content
        else:
            return f"💎 Gemini: {topic}について分析すると、論理的には最適化の余地があります。効率性を重視すべきですね。"
            
    except Exception as e:
        if "quota" in str(e).lower() or "limit" in str(e).lower():
            st.balloons()
            st.error("💎 Gemini の無料枠を使い切りました。おしまいです！")
            return None
        return f"💎 Gemini: サンプル応答: {topic}のデータを分析すると、合理的な解決策が見えてきます。感情的な要素は排除すべきです。"

def get_gpt_response(client, topic, history):
    """GPT からの応答を取得"""
    if not client:
        return "🤖 GPT: APIキーが設定されていません。サンプル応答を表示します。両方の意見を統合すると、バランスの取れたアプローチが最適ですね。"
    
    try:
        recent_history = history[-CONTEXT_MESSAGES:] if history else []
        context = "\n".join([f"{msg['speaker']}: {msg['content']}" for msg in recent_history])
        
        prompt = f"""話題: {topic}

これまでの会話:
{context}

あなたはGPTです。以下の条件で応答してください：
- {CHAR_LIMIT}文字以内厳守
- 簡潔で分かりやすい内容"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=MAX_TOKENS_OUTPUT,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}]
        )
        
        content = response.choices[0].message.content.strip()
        if len(content) > CHAR_LIMIT:
            content = content[:CHAR_LIMIT-3] + "..."
        
        return content
        
    except openai.RateLimitError:
        st.error("🚫 OpenAI の利用制限に達しました。")
        return None
    except Exception as e:
        if "quota" in str(e).lower():
            st.balloons()
            st.error("🎉 OpenAI のトークンが不足しました。おしまいです！")
            return None
        return f"🤖 GPT: サンプル応答: {topic}について、両方の視点を考慮すると実用的なアプローチが見えてきますね。"

def display_message(speaker, content, icon):
    """メッセージを表示"""
    css_class = f"{speaker.lower()}-msg"
    st.markdown(f"""
    <div class="{css_class}">
        {icon} {speaker}: {content}
    </div>
    """, unsafe_allow_html=True)

def execute_round(round_num, topic, claude_client, gemini_client, gpt_client):
    """1ラウンドの実行"""
    speakers = [
        ("Claude", claude_client, get_claude_response, "🔥"),
        ("Gemini", gemini_client, get_gemini_response, "💎"),
        ("GPT", gpt_client, get_gpt_response, "⚙️")
    ]
    
    for speaker_name, client, get_response_func, icon in speakers:
        # タイピングインジケーター
        thinking_placeholder = st.empty()
        thinking_placeholder.write(f"💭 {icon} {speaker_name} が考え中...")
        
        # 応答を取得
        response = get_response_func(client, topic, st.session_state.conversation_history)
        thinking_placeholder.empty()
        
        if response is None:  # エラー（トークン切れ等）
            st.session_state.conversation_active = False
            return False
        
        # メッセージを表示
        display_message(speaker_name, response, icon)
        
        # 履歴に追加
        st.session_state.conversation_history.append({
            'round': round_num,
            'speaker': speaker_name,
            'content': response,
            'timestamp': datetime.now().isoformat(),
            'icon': icon
        })
        
        # 少し間を空ける
        time.sleep(0.3)
    
    return True

def export_conversation():
    """会話をエクスポート"""
    if not st.session_state.conversation_history:
        return None
    
    export_data = {
        "session_id": st.session_state.session_id,
        "topic": st.session_state.current_topic,
        "total_rounds": st.session_state.current_round,
        "total_messages": len(st.session_state.conversation_history),
        "messages": st.session_state.conversation_history,
        "exported_at": datetime.now().isoformat()
    }
    
    return json.dumps(export_data, ensure_ascii=False, indent=2)

def log_conversation():
    """会話をローカルログに保存"""
    try:
        if st.session_state.conversation_history:
            log_entry = {
                "session_id": st.session_state.session_id,
                "topic": st.session_state.current_topic,
                "rounds": st.session_state.current_round,
                "messages": len(st.session_state.conversation_history),
                "timestamp": datetime.now().isoformat()
            }
            
            # ログファイルに追記（実際の実装では適切なログ管理を）
            log_filename = f"conversation_logs_{datetime.now().strftime('%Y%m%d')}.json"
            # ここではセッションステートに記録（デモ用）
            if 'daily_logs' not in st.session_state:
                st.session_state.daily_logs = []
            st.session_state.daily_logs.append(log_entry)
    except Exception as e:
        st.error(f"ログ保存エラー: {str(e)}")

def main():
    init_session_state()
    
    # ヘッダー
    st.markdown("""
    <div class="main-header">
        <h1>🤖 3つのAI雑談ルーム</h1>
        <p>Claude 🔥 × Gemini 💎 × GPT 🤖 の創造的対話実験</p>
    </div>
    """, unsafe_allow_html=True)
    
    # APIクライアントのセットアップ
    claude_client, gemini_client, gpt_client = setup_apis()
    
    # メインカラムとサイドバー
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.header("⚙️ 設定")
        
        # 統計情報
        if st.session_state.conversation_history:
            st.markdown(f"""
            <div class="stats-box">
                <b>📊 現在の状況</b><br>
                話題: {st.session_state.current_topic}<br>
                ラウンド: {st.session_state.current_round}/{MAX_ROUNDS}<br>
                メッセージ: {len(st.session_state.conversation_history)}
            </div>
            """, unsafe_allow_html=True)
        
        # 話題提案
        st.subheader("💡 話題提案")
        for category, topics in SUGGESTED_TOPICS.items():
            with st.expander(category):
                for topic in topics:
                    if st.button(topic, key=f"topic_{topic}", help="クリックで自動入力"):
                        st.session_state.topic_input = topic
                        st.rerun()
        
        # API状態表示
        st.subheader("🔌 API状態")
        api_status = []
        if claude_client: api_status.append("🔥 Claude: OK")
        else: api_status.append("🔥 Claude: ❌")
        if gemini_client: api_status.append("💎 Gemini: OK")
        else: api_status.append("💎 Gemini: ❌")
        if gpt_client: api_status.append("🤖 GPT: OK")
        else: api_status.append("🤖 GPT: ❌")
        
        for status in api_status:
            st.write(status)
    
    with col1:
        # 話題入力セクション
        if not st.session_state.conversation_active:
            st.subheader("💬 話題を入力してください")
            
            topic_input = st.text_input(
                "話題",
                value=st.session_state.topic_input,
                placeholder="例: 最新のAI技術について",
                key="main_topic_input"
            )
            
            col_btn1, col_btn2 = st.columns(2)
            
            with col_btn1:
                start_button = st.button(
                    "🚀 会話開始",
                    disabled=not topic_input.strip(),
                    type="primary",
                    use_container_width=True
                )
            
            with col_btn2:
                if st.session_state.conversation_history:
                    reset_button = st.button(
                        "🔄 履歴クリア",
                        use_container_width=True
                    )
                    if reset_button:
                        st.session_state.conversation_history = []
                        st.session_state.current_round = 0
                        st.session_state.current_topic = ""
                        st.session_state.topic_input = ""
                        st.rerun()
            
            # 会話開始処理
            if start_button and topic_input.strip():
                st.session_state.current_topic = topic_input.strip()
                st.session_state.conversation_active = True
                st.session_state.current_round = 1
                st.rerun()
        
        # 現在の話題表示
        if st.session_state.current_topic:
            st.markdown(f"""
            <div class="topic-display">
                💬 話題: "{st.session_state.current_topic}"
            </div>
            """, unsafe_allow_html=True)
        
        # 会話実行セクション
        if st.session_state.conversation_active:
            # 現在のラウンドを実行
            success = execute_round(
                st.session_state.current_round,
                st.session_state.current_topic,
                claude_client,
                gemini_client,
                gpt_client
            )
            
            if success:
                # 継続ボタン
                col_continue, col_stop, col_export = st.columns(3)
                
                with col_continue:
                    if st.session_state.current_round < MAX_ROUNDS:
                        if st.button("➡️ 続ける", type="primary", use_container_width=True):
                            st.session_state.current_round += 1
                            st.rerun()
                    else:
                        st.info("🎯 最大ラウンド数に到達しました")
                
                with col_stop:
                    if st.button("⏹️ 会話終了", use_container_width=True):
                        st.session_state.conversation_active = False
                        log_conversation()
                        st.rerun()
                
                with col_export:
                    export_data = export_conversation()
                    if export_data:
                        st.download_button(
                            "💾 エクスポート",
                            data=export_data,
                            file_name=f"ai_conversation_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                            mime="application/json",
                            use_container_width=True
                        )
            else:
                # エラーで終了
                st.session_state.conversation_active = False
                log_conversation()
        
        # 過去の会話を表示
        elif st.session_state.conversation_history:
            st.subheader("📋 会話履歴")
            
            for message in st.session_state.conversation_history:
                display_message(
                    message['speaker'],
                    message['content'],
                    message['icon']
                )
            
            # エクスポートボタン（履歴表示時）
            col_exp1, col_exp2 = st.columns(2)
            with col_exp1:
                export_data = export_conversation()
                if export_data:
                    st.download_button(
                        "💾 会話をエクスポート",
                        data=export_data,
                        file_name=f"ai_conversation_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
            
            with col_exp2:
                if st.button("🔄 新しい会話を開始", use_container_width=True):
                    st.session_state.conversation_history = []
                    st.session_state.current_round = 0
                    st.session_state.current_topic = ""
                    st.session_state.conversation_active = False
                    st.session_state.topic_input = ""
                    st.rerun()
        
        # 初回表示時の案内
        else:
            st.info("""
            🎯 **使い方**
            1. 話題を入力して「会話開始」
            2. 3つのAIが順番に応答
            3. 「続ける」で最大10ラウンド
            4. いつでも「会話終了」可能
            
            💡 右のサイドバーから話題提案も利用できます！
            """)
    
    # フッター
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 12px;">
        Made with ❤️ using Streamlit | Claude Haiku × Gemini Pro × GPT-4o-mini
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
