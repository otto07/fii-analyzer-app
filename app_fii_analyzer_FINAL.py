"""
Sistema Completo de Análise de FIIs - VERSÃO FINAL
Versão: 4.1 - Score Completo + Comparativo + Carteira

RECURSOS:
✓ Score: 60% Fundamentalista + 40% Técnico
✓ Análise Técnica: RSI, MACD, Médias Móveis
✓ Análise Individual detalhada
✓ Comparativo de múltiplos FIIs
✓ Carteira sugerida com simulações
✓ Gráficos interativos
✓ Exportação CSV

Execute: python -m streamlit run app_fii_analyzer_FINAL.py
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Análise de FIIs v4.1 - Completo",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .score-breakdown {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .technical-score {
        background: linear-gradient(135deg, #fa8231 0%, #f76b1c 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .fundamental-score {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .buy-signal {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
    }
    .sell-signal {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
    }
    .neutral-signal {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

class FIIAnalyzerApp:

    def __init__(self):
        self.initialize_session_state()

    def initialize_session_state(self):
        if 'analyzed_fiis' not in st.session_state:
            st.session_state.analyzed_fiis = {}
        if 'portfolio_value' not in st.session_state:
            st.session_state.portfolio_value = 50000
        if 'risk_tolerance' not in st.session_state:
            st.session_state.risk_tolerance = 'Moderado'
        if 'current_view' not in st.session_state:
            st.session_state.current_view = 'home'
        if 'selected_fii' not in st.session_state:
            st.session_state.selected_fii = None

    def get_fii_data(self, ticker):
        try:
            fii = yf.Ticker(ticker)

            try:
                info = fii.info
                if not info or len(info) < 5:
                    info = {}
            except:
                info = {}

            hist = fii.history(period='1y')

            if hist.empty:
                st.error(f"❌ {ticker}: Sem dados históricos")
                return None

            try:
                dividends = fii.dividends
                if not dividends.empty:
                    dividends = dividends.tail(12)
                else:
                    dividends = pd.Series()
            except:
                dividends = pd.Series()

            volatility = hist['Close'].pct_change().std() * np.sqrt(252) * 100
            returns_12m = ((hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1) * 100

            preco_atual = (
                info.get('regularMarketPrice') or 
                info.get('currentPrice') or 
                info.get('previousClose') or
                hist['Close'].iloc[-1]
            )

            data = {
                'ticker': ticker,
                'nome': info.get('longName', ticker.replace('.SA', '')),
                'preco_atual': float(preco_atual),
                'dividend_yield': float(info.get('dividendYield', 0) * 100) if info.get('dividendYield') else 0,
                'p_vp': float(info.get('priceToBook', 1.0) or 1.0),
                'volume_medio': float(info.get('averageVolume', hist['Volume'].mean()) or hist['Volume'].mean()),
                'volatilidade': float(volatility),
                'retorno_12m': float(returns_12m),
                'max_52w': float(info.get('fiftyTwoWeekHigh', hist['High'].max()) or hist['High'].max()),
                'min_52w': float(info.get('fiftyTwoWeekLow', hist['Low'].min()) or hist['Low'].min()),
                'num_dividendos': len(dividends),
                'total_dividendos_12m': float(dividends.sum()) if not dividends.empty else 0,
                'historico': hist,
                'dividendos': dividends
            }

            return data

        except Exception as e:
            st.error(f"❌ Erro: {str(e)}")
            return None

    def calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not rsi.empty else 50

    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_histogram = macd - macd_signal
        return {
            'macd': macd.iloc[-1] if not macd.empty else 0,
            'signal': macd_signal.iloc[-1] if not macd_signal.empty else 0,
            'histogram': macd_histogram.iloc[-1] if not macd_histogram.empty else 0
        }

    def calculate_technical_score(self, fii_data):
        hist = fii_data['historico']
        prices = hist['Close']
        score_tecnico = 0
        detalhes = {}

        # RSI (15 pontos)
        rsi = self.calculate_rsi(prices)
        detalhes['rsi'] = rsi

        if 40 <= rsi <= 60:
            score_tecnico += 15
            detalhes['rsi_status'] = 'Neutro/Ideal'
        elif 30 <= rsi < 40 or 60 < rsi <= 70:
            score_tecnico += 10
            detalhes['rsi_status'] = 'Moderado'
        elif rsi < 30:
            score_tecnico += 12
            detalhes['rsi_status'] = 'Sobrevendido (Oportunidade)'
        elif rsi > 70:
            score_tecnico += 5
            detalhes['rsi_status'] = 'Sobrecomprado (Cuidado)'

        # Médias Móveis (15 pontos)
        sma20 = prices.rolling(window=20).mean().iloc[-1]
        sma50 = prices.rolling(window=50).mean().iloc[-1] if len(prices) >= 50 else sma20
        preco_atual = prices.iloc[-1]

        detalhes['sma20'] = sma20
        detalhes['sma50'] = sma50
        detalhes['preco_atual'] = preco_atual

        if preco_atual > sma20 and preco_atual > sma50:
            score_tecnico += 15
            detalhes['ma_status'] = 'Acima SMA20 e SMA50 (Forte)'
        elif preco_atual > sma20:
            score_tecnico += 10
            detalhes['ma_status'] = 'Acima SMA20 (Moderado)'
        elif preco_atual > sma50:
            score_tecnico += 7
            detalhes['ma_status'] = 'Entre SMA20 e SMA50'
        else:
            score_tecnico += 3
            detalhes['ma_status'] = 'Abaixo das Médias (Fraco)'

        # MACD (10 pontos)
        macd_data = self.calculate_macd(prices)
        detalhes['macd'] = macd_data['macd']
        detalhes['macd_signal'] = macd_data['signal']
        detalhes['macd_histogram'] = macd_data['histogram']

        if macd_data['histogram'] > 0:
            score_tecnico += 10
            detalhes['macd_status'] = 'Bullish (Positivo)'
        elif abs(macd_data['histogram']) < 0.5:
            score_tecnico += 5
            detalhes['macd_status'] = 'Neutro'
        else:
            score_tecnico += 2
            detalhes['macd_status'] = 'Bearish (Negativo)'

        return score_tecnico, detalhes

    def calculate_fundamental_score(self, fii_data):
        score_fundamental = 0
        detalhes = {}

        # DY (20 pontos)
        dy = fii_data['dividend_yield']
        detalhes['dividend_yield'] = dy
        if dy > 10:
            score_fundamental += 20
            detalhes['dy_status'] = 'Excelente (>10%)'
        elif dy > 8:
            score_fundamental += 15
            detalhes['dy_status'] = 'Muito Bom (>8%)'
        elif dy > 6:
            score_fundamental += 10
            detalhes['dy_status'] = 'Bom (>6%)'
        elif dy > 4:
            score_fundamental += 5
            detalhes['dy_status'] = 'Regular (>4%)'
        else:
            detalhes['dy_status'] = 'Baixo (≤4%)'

        # P/VP (15 pontos)
        pvp = fii_data['p_vp']
        detalhes['p_vp'] = pvp
        if 0.70 <= pvp <= 0.95:
            score_fundamental += 15
            detalhes['pvp_status'] = 'Ideal (0.70-0.95)'
        elif 0.95 < pvp <= 1.05:
            score_fundamental += 10
            detalhes['pvp_status'] = 'Bom (0.95-1.05)'
        elif pvp < 0.70:
            score_fundamental += 5
            detalhes['pvp_status'] = 'Muito Baixo (<0.70)'
        else:
            detalhes['pvp_status'] = 'Alto (>1.05)'

        # Volume (10 pontos)
        volume = fii_data['volume_medio']
        detalhes['volume'] = volume
        if volume > 1000000:
            score_fundamental += 10
            detalhes['volume_status'] = 'Alta (>1M)'
        elif volume > 500000:
            score_fundamental += 7
            detalhes['volume_status'] = 'Boa (>500k)'
        elif volume > 100000:
            score_fundamental += 4
            detalhes['volume_status'] = 'Regular (>100k)'
        else:
            detalhes['volume_status'] = 'Baixa (≤100k)'

        # Volatilidade (10 pontos)
        vol = fii_data['volatilidade']
        detalhes['volatilidade'] = vol
        if vol < 15:
            score_fundamental += 10
            detalhes['vol_status'] = 'Muito Baixa (<15%)'
        elif vol < 25:
            score_fundamental += 7
            detalhes['vol_status'] = 'Baixa (<25%)'
        elif vol < 35:
            score_fundamental += 4
            detalhes['vol_status'] = 'Moderada (<35%)'
        else:
            detalhes['vol_status'] = 'Alta (≥35%)'

        # Retorno 12m (5 pontos)
        ret = fii_data['retorno_12m']
        detalhes['retorno_12m'] = ret
        if ret > 10:
            score_fundamental += 5
            detalhes['ret_status'] = 'Excelente (>10%)'
        elif ret > 0:
            score_fundamental += 3
            detalhes['ret_status'] = 'Positivo (>0%)'
        else:
            detalhes['ret_status'] = 'Negativo (≤0%)'

        return score_fundamental, detalhes

    def calculate_total_score(self, fii_data):
        score_fundamental, detalhes_fund = self.calculate_fundamental_score(fii_data)
        score_tecnico, detalhes_tec = self.calculate_technical_score(fii_data)
        score_total = score_fundamental + score_tecnico
        return {
            'score_total': score_total,
            'score_fundamental': score_fundamental,
            'score_tecnico': score_tecnico,
            'detalhes_fundamental': detalhes_fund,
            'detalhes_tecnico': detalhes_tec
        }

    def get_recommendation(self, score_total):
        if score_total >= 80:
            return {'action': 'COMPRA FORTE', 'color': 'green', 
                   'message': '🎯 Excelente oportunidade! Fundamentos e técnica favoráveis.'}
        elif score_total >= 65:
            return {'action': 'COMPRA', 'color': 'lightgreen',
                   'message': '✅ Bons indicadores. Momento adequado para compra.'}
        elif score_total >= 50:
            return {'action': 'NEUTRO', 'color': 'yellow',
                   'message': '⚠️ Indicadores moderados. Considere aguardar melhores sinais.'}
        elif score_total >= 35:
            return {'action': 'CAUTELA', 'color': 'orange',
                   'message': '⏳ Indicadores fracos. Aguarde melhores condições.'}
        else:
            return {'action': 'EVITAR', 'color': 'red',
                   'message': '❌ Indicadores desfavoráveis. Não recomendado.'}

    def calculate_support_resistance(self, hist_data, n=2):
        highs = hist_data['High'].values
        lows = hist_data['Low'].values
        resistance_levels = []
        support_levels = []

        for i in range(n, len(highs) - n):
            if all(highs[i] > highs[i-j] for j in range(1, n+1)) and                all(highs[i] > highs[i+j] for j in range(1, n+1)):
                resistance_levels.append(highs[i])

        for i in range(n, len(lows) - n):
            if all(lows[i] < lows[i-j] for j in range(1, n+1)) and                all(lows[i] < lows[i+j] for j in range(1, n+1)):
                support_levels.append(lows[i])

        def consolidate_levels(levels, threshold=0.02):
            if not levels:
                return []
            levels = sorted(levels)
            consolidated = [levels[0]]
            for level in levels[1:]:
                if (level - consolidated[-1]) / consolidated[-1] > threshold:
                    consolidated.append(level)
            return consolidated

        return consolidate_levels(support_levels), consolidate_levels(resistance_levels)

    def calculate_entry_price(self, fii_data):
        current_price = fii_data['preco_atual']
        hist = fii_data['historico']
        support_levels, resistance_levels = self.calculate_support_resistance(hist)

        entry_price = current_price
        entry_type = "Preço Atual"
        valid_supports = [s for s in support_levels if s < current_price * 0.98]

        if valid_supports:
            nearest_support = max(valid_supports)
            entry_price = nearest_support * 1.01
            entry_type = f"Suporte (R$ {nearest_support:.2f}) + 1%"
        else:
            min_30d = hist['Low'].tail(30).min()
            if min_30d < current_price * 0.95:
                entry_price = min_30d * 1.02
                entry_type = f"Mínimo 30d (R$ {min_30d:.2f}) + 2%"
            else:
                entry_price = current_price * 0.98
                entry_type = "Preço Atual - 2%"

        stop_loss = entry_price * 0.95
        target_price = entry_price * 1.10
        valid_resistances = [r for r in resistance_levels if r > current_price * 1.02]
        if valid_resistances:
            target_price = min(valid_resistances)

        return {
            'entry_price': entry_price,
            'entry_type': entry_type,
            'stop_loss': stop_loss,
            'target_price': target_price,
            'support_levels': support_levels,
            'resistance_levels': resistance_levels,
            'discount_pct': ((current_price - entry_price) / current_price) * 100
        }

    def calculate_position_size(self, fii_data, entry_data, portfolio_value, risk_tolerance):
        entry_price = entry_data['entry_price']
        stop_loss = entry_data['stop_loss']
        risk_per_trade = {'Conservador': 0.01, 'Moderado': 0.02, 'Agressivo': 0.03}
        risk_pct = risk_per_trade.get(risk_tolerance, 0.02)
        risk_amount = portfolio_value * risk_pct
        risk_per_share = entry_price - stop_loss

        quantity_by_risk = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
        max_investment = portfolio_value * 0.20
        max_quantity = int(max_investment / entry_price)
        final_quantity = min(quantity_by_risk, max_quantity)
        if final_quantity == 0 and portfolio_value >= entry_price:
            final_quantity = 1

        total_investment = final_quantity * entry_price
        monthly_dividend = 0
        annual_dividend = 0
        if fii_data['dividend_yield'] > 0 and final_quantity > 0:
            annual_dividend = (total_investment * (fii_data['dividend_yield'] / 100))
            monthly_dividend = annual_dividend / 12

        potential_gain = (entry_data['target_price'] - entry_price) * final_quantity
        potential_loss = (entry_price - stop_loss) * final_quantity

        return {
            'quantity': final_quantity,
            'total_investment': total_investment,
            'portfolio_pct': (total_investment / portfolio_value) * 100 if portfolio_value > 0 else 0,
            'monthly_dividend': monthly_dividend,
            'annual_dividend': annual_dividend,
            'potential_gain': potential_gain,
            'potential_loss': potential_loss,
            'risk_reward_ratio': potential_gain / potential_loss if potential_loss > 0 else 0
        }

app = FIIAnalyzerApp()

# CABEÇALHO
st.markdown('<div class="main-header">📊 Análise de FIIs v4.1 - Completo (60% Fund + 40% Téc)</div>', 
            unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.header("⚙️ Configurações")

    portfolio_value = st.number_input("💰 Portfólio (R$)", 1000, 10000000, 
                                     st.session_state.portfolio_value, 1000)
    st.session_state.portfolio_value = portfolio_value

    risk_tolerance = st.select_slider("🎯 Perfil", ['Conservador', 'Moderado', 'Agressivo'],
                                     value=st.session_state.risk_tolerance)
    st.session_state.risk_tolerance = risk_tolerance

    st.markdown("---")
    st.header("📝 Busca Rápida")

    search_col1, search_col2 = st.columns([3,1])
    with search_col1:
        ticker_input = st.text_input("🔍", placeholder="Ex: KNRI11", label_visibility="collapsed")
    with search_col2:
        search_btn = st.button("🔍", use_container_width=True)

    if search_btn and ticker_input:
        ticker = ticker_input.strip().upper()
        if not ticker.endswith('.SA'):
            ticker += '.SA'

        with st.spinner(f'Analisando {ticker}...'):
            fii_data = app.get_fii_data(ticker)
            if fii_data:
                st.session_state.analyzed_fiis[ticker] = fii_data
                st.session_state.selected_fii = ticker
                st.session_state.current_view = 'analysis'
                st.success(f"✅ {ticker}")
                time.sleep(0.3)
                st.rerun()

    if st.session_state.analyzed_fiis:
        st.markdown("---")
        st.subheader(f"📋 Analisados ({len(st.session_state.analyzed_fiis)})")

        for ticker in list(st.session_state.analyzed_fiis.keys()):
            col1, col2 = st.columns([3,1])
            with col1:
                if st.button(ticker.replace('.SA',''), key=f"nav_{ticker}", use_container_width=True):
                    st.session_state.selected_fii = ticker
                    st.session_state.current_view = 'analysis'
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"del_{ticker}"):
                    del st.session_state.analyzed_fiis[ticker]
                    if st.session_state.selected_fii == ticker:
                        st.session_state.selected_fii = None
                        st.session_state.current_view = 'home'
                    st.rerun()

        if st.button("🗑️ Limpar Todos"):
            st.session_state.analyzed_fiis = {}
            st.session_state.current_view = 'home'
            st.rerun()

    st.markdown("---")
    if st.button("🏠 Início", use_container_width=True, type="secondary"):
        st.session_state.current_view = 'home'
        st.rerun()
    
# ANÁLISE MÚLTIPLA
    st.markdown("---")
    st.subheader("⚡ Análise Múltipla")
    st.caption("Analise vários FIIs de uma vez")
    
    quick_fiis = st.text_area(
        "FIIs (um por linha)",
        placeholder="KNRI11\nXPML11\nHGLG11\nBTLG11\nVISC11",
        height=100,
        key="quick_fiis"
    )
    
    if st.button("⚡ Analisar Todos", type="primary", key="analyze_all"):
        if quick_fiis and quick_fiis.strip():
            tickers = [t.strip().upper() for t in quick_fiis.split('\n') if t.strip()]
            
            if tickers:
                progress_bar = st.progress(0)
                status_text = st.empty()
                success_count = 0
                
                for idx, ticker in enumerate(tickers):
                    if not ticker.endswith('.SA'):
                        ticker += '.SA'
                    
                    status_text.text(f"Analisando {ticker}... ({idx+1}/{len(tickers)})")
                    fii_data = app.get_fii_data(ticker)
                    if fii_data:
                        st.session_state.analyzed_fiis[ticker] = fii_data
                        success_count += 1
                    
                    progress_bar.progress((idx + 1) / len(tickers))
                    time.sleep(0.3)
                
                status_text.empty()
                progress_bar.empty()
                
                if success_count > 0:
                    st.success(f"✅ {success_count}/{len(tickers)} FIIs analisados!")
                    time.sleep(1)
                    st.rerun()

# ÁREA PRINCIPAL
if st.session_state.current_view == 'home' or not st.session_state.analyzed_fiis:
    st.subheader("💡 FIIs Populares")

    popular_fiis = ['KNRI11', 'XPML11', 'HGLG11', 'BTLG11',
                   'VISC11', 'MXRF11', 'BCFF11', 'KNCR11']

    cols = st.columns(4)
    for idx, fii in enumerate(popular_fiis):
        with cols[idx % 4]:
            if st.button(fii, key=f"pop_{fii}", use_container_width=True):
                ticker = fii + '.SA'
                with st.spinner(f'Analisando {ticker}...'):
                    fii_data = app.get_fii_data(ticker)
                    if fii_data:
                        st.session_state.analyzed_fiis[ticker] = fii_data
                        st.session_state.selected_fii = ticker
                        st.session_state.current_view = 'analysis'
                        time.sleep(0.3)
                        st.rerun()

else:
    selected_ticker = st.session_state.selected_fii

    if selected_ticker and selected_ticker in st.session_state.analyzed_fiis:
        fii_data = st.session_state.analyzed_fiis[selected_ticker]
        scores = app.calculate_total_score(fii_data)
        entry_data = app.calculate_entry_price(fii_data)
        recommendation = app.get_recommendation(scores['score_total'])
        position = app.calculate_position_size(fii_data, entry_data, portfolio_value, risk_tolerance)

        # TABS
        tab1, tab2, tab3 = st.tabs(["📊 Análise Individual", "📈 Comparativo", "💼 Carteira"])


        with tab1:
            # RECOMENDAÇÃO
            rec_class = {'COMPRA FORTE': 'buy-signal', 'COMPRA': 'buy-signal',
                        'NEUTRO': 'neutral-signal', 'CAUTELA': 'neutral-signal',
                        'EVITAR': 'sell-signal'}

            st.markdown(f"""
            <div class="{rec_class.get(recommendation['action'], 'neutral-signal')}">
                <h2 style='margin:0'>{recommendation['action']}</h2>
                <p style='margin:0.5rem 0'>{recommendation['message']}</p>
                <h3 style='margin:0'>Score Total: {scores['score_total']}/100</h3>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")

            # BREAKDOWN DO SCORE
            st.subheader("📊 Detalhamento do Score")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"""
                <div class="fundamental-score">
                    <h3 style='margin:0'>Score Fundamentalista</h3>
                    <h2 style='margin:0.5rem 0'>{scores['score_fundamental']}/60</h2>
                    <p style='margin:0; font-size:0.9rem'>60% do score total</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="technical-score">
                    <h3 style='margin:0'>Score Técnico</h3>
                    <h2 style='margin:0.5rem 0'>{scores['score_tecnico']}/40</h2>
                    <p style='margin:0; font-size:0.9rem'>40% do score total</p>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                perc_fund = (scores['score_fundamental'] / 60) * 100
                perc_tec = (scores['score_tecnico'] / 40) * 100
                st.markdown(f"""
                <div class="score-breakdown">
                    <h4 style='margin:0'>Percentuais</h4>
                    <p style='margin:0.3rem 0'>Fundamentalista: {perc_fund:.1f}%</p>
                    <p style='margin:0.3rem 0'>Técnico: {perc_tec:.1f}%</p>
                    <p style='margin:0.3rem 0; font-weight:bold'>Total: {scores['score_total']}/100</p>
                </div>
                """, unsafe_allow_html=True)

            # DETALHES FUNDAMENTALISTAS
            with st.expander("📈 Detalhes da Análise Fundamentalista", expanded=True):
                det_f = scores['detalhes_fundamental']
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Dividend Yield", f"{det_f['dividend_yield']:.2f}%")
                    st.caption(det_f['dy_status'])
                    st.metric("P/VP", f"{det_f['p_vp']:.2f}")
                    st.caption(det_f['pvp_status'])

                with col2:
                    st.metric("Volume Médio", f"{det_f['volume']:,.0f}")
                    st.caption(det_f['volume_status'])
                    st.metric("Volatilidade", f"{det_f['volatilidade']:.1f}%")
                    st.caption(det_f['vol_status'])

                with col3:
                    st.metric("Retorno 12m", f"{det_f['retorno_12m']:+.1f}%")
                    st.caption(det_f['ret_status'])

            # DETALHES TÉCNICOS
            with st.expander("📉 Detalhes da Análise Técnica", expanded=True):
                det_t = scores['detalhes_tecnico']
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("RSI (14)", f"{det_t['rsi']:.1f}")
                    st.caption(det_t['rsi_status'])
                    if det_t['rsi'] < 30:
                        st.success("🟢 Sobrevendido - Oportunidade")
                    elif det_t['rsi'] > 70:
                        st.error("🔴 Sobrecomprado - Cuidado")
                    else:
                        st.info("🟡 Zona Neutra")

                with col2:
                    st.metric("Preço vs SMA20", 
                             f"{((det_t['preco_atual']/det_t['sma20'])-1)*100:+.1f}%")
                    st.caption(f"SMA20: R$ {det_t['sma20']:.2f}")
                    st.metric("Preço vs SMA50",
                             f"{((det_t['preco_atual']/det_t['sma50'])-1)*100:+.1f}%")
                    st.caption(f"SMA50: R$ {det_t['sma50']:.2f}")
                    st.caption(det_t['ma_status'])

                with col3:
                    st.metric("MACD", f"{det_t['macd']:.4f}")
                    st.metric("Sinal", f"{det_t['macd_signal']:.4f}")
                    st.metric("Histograma", f"{det_t['macd_histogram']:.4f}")
                    st.caption(det_t['macd_status'])
                    if det_t['macd_histogram'] > 0:
                        st.success("🟢 Sinal de Compra")
                    else:
                        st.error("🔴 Sinal de Venda")

            st.markdown("---")

            # ESTRATÉGIA
            st.subheader("🎯 Estratégia de Entrada")
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Entrada Sugerida", f"R$ {entry_data['entry_price']:.2f}",
                         f"{entry_data['discount_pct']:+.1f}% vs atual")
                st.caption(f"📍 {entry_data['entry_type']}")
                st.metric("Stop Loss", f"R$ {entry_data['stop_loss']:.2f}", "-5.0%")
                st.metric("Alvo", f"R$ {entry_data['target_price']:.2f}",
                         f"+{((entry_data['target_price']/entry_data['entry_price'])-1)*100:.1f}%")

            with col2:
                st.metric("Quantidade", f"{position['quantity']} cotas")
                st.metric("Investimento", f"R$ {position['total_investment']:.2f}",
                         f"{position['portfolio_pct']:.1f}% portfólio")
                st.metric("Div. Mensal", f"R$ {position['monthly_dividend']:.2f}",
                         f"R$ {position['annual_dividend']:.2f}/ano")

            st.markdown("---")

            # RISCO/RETORNO
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Ganho Potencial", f"R$ {position['potential_gain']:.2f}",
                         f"+{(position['potential_gain']/position['total_investment'])*100:.1f}%")
            with col2:
                st.metric("Perda Máxima", f"R$ {position['potential_loss']:.2f}",
                         f"-{(position['potential_loss']/position['total_investment'])*100:.1f}%")
            with col3:
                st.metric("Ratio R/R", f"{position['risk_reward_ratio']:.2f}:1",
                         "✅" if position['risk_reward_ratio'] >= 2 else "⚠️")

            st.markdown("---")

            # GRÁFICO DE CANDLESTICK
            st.subheader("📈 Gráfico de Preços (Candlestick)")

            hist = fii_data['historico']

            # Criar gráfico de candlestick
            fig_candle = go.Figure(data=[go.Candlestick(
                x=hist.index,
                open=hist['Open'],
                high=hist['High'],
                low=hist['Low'],
                close=hist['Close'],
                name='Preço'
            )])

            # Adicionar médias móveis
            hist['SMA20'] = hist['Close'].rolling(window=20).mean()
            hist['SMA50'] = hist['Close'].rolling(window=50).mean()

            fig_candle.add_trace(go.Scatter(
                x=hist.index,
                y=hist['SMA20'],
                mode='lines',
                name='SMA 20',
                line=dict(color='orange', width=1)
            ))

            fig_candle.add_trace(go.Scatter(
                x=hist.index,
                y=hist['SMA50'],
                mode='lines',
                name='SMA 50',
                line=dict(color='blue', width=1)
            ))

            # Adicionar linhas de suporte e resistência
            if entry_data['support_levels']:
                for support in entry_data['support_levels'][-3:]:  # Últimos 3 suportes
                    fig_candle.add_hline(
                        y=support, 
                        line_dash="dash", 
                        line_color="green",
                        annotation_text=f"Suporte: R$ {support:.2f}",
                        annotation_position="right"
                    )

            if entry_data['resistance_levels']:
                for resistance in entry_data['resistance_levels'][-3:]:  # Últimas 3 resistências
                    fig_candle.add_hline(
                        y=resistance, 
                        line_dash="dash", 
                        line_color="red",
                        annotation_text=f"Resistência: R$ {resistance:.2f}",
                        annotation_position="right"
                    )

            # Adicionar linha de entrada sugerida
            fig_candle.add_hline(
                y=entry_data['entry_price'], 
                line_dash="dot", 
                line_color="yellow", 
                line_width=2,
                annotation_text=f"Entrada: R$ {entry_data['entry_price']:.2f}",
                annotation_position="left"
            )

            fig_candle.update_layout(
                title=f"{fii_data['ticker'].replace('.SA', '')} - Últimos 12 Meses",
                yaxis_title="Preço (R$)",
                xaxis_title="Data",
                height=500,
                xaxis_rangeslider_visible=False,
                hovermode='x unified'
            )

            st.plotly_chart(fig_candle, use_container_width=True)

            st.markdown("---")

            # ORDEM SUGERIDA DE COMPRA
            st.subheader("📋 Ordem de Compra Sugerida")

            ordem_col1, ordem_col2 = st.columns(2)

            with ordem_col1:
                st.markdown(f"""
                **📊 Dados da Ordem:**
                - **Ticker**: {fii_data['ticker'].replace('.SA', '')}
                - **Tipo**: Ordem Limitada
                - **Operação**: COMPRA
                - **Quantidade**: {position['quantity']} cotas
                - **Preço de Entrada**: R$ {entry_data['entry_price']:.2f}
                - **Validade**: 30 dias (GTC - Good Till Cancelled)
                """)

                st.info(f"""
                💡 **Estratégia**: {entry_data['entry_type']}

                Aguardar o preço atingir R$ {entry_data['entry_price']:.2f} 
                (desconto de {entry_data['discount_pct']:.1f}% vs preço atual)
                """)

            with ordem_col2:
                st.markdown(f"""
                **🎯 Gestão de Risco:**
                - **Stop Loss**: R$ {entry_data['stop_loss']:.2f} (-5%)
                - **Take Profit**: R$ {entry_data['target_price']:.2f} 
                  (+{((entry_data['target_price']/entry_data['entry_price'])-1)*100:.1f}%)
                - **Ratio Risco/Retorno**: {position['risk_reward_ratio']:.2f}:1
                - **Perda Máxima**: R$ {position['potential_loss']:.2f}
                - **Ganho Esperado**: R$ {position['potential_gain']:.2f}
                """)

                if position['risk_reward_ratio'] >= 2:
                    st.success("✅ Ratio favorável (≥ 2:1)")
                else:
                    st.warning("⚠️ Ratio desfavorável (< 2:1)")

            # Botão para copiar ordem
            ordem_texto = f"""ORDEM DE COMPRA - {fii_data['ticker'].replace('.SA', '')}

Tipo: Ordem Limitada
Operação: COMPRA
Quantidade: {position['quantity']} cotas
Preço Limite: R$ {entry_data['entry_price']:.2f}
Validade: 30 dias

STOP LOSS: R$ {entry_data['stop_loss']:.2f}
TAKE PROFIT: R$ {entry_data['target_price']:.2f}

Investimento Total: R$ {position['total_investment']:.2f}
Dividendo Mensal Esperado: R$ {position['monthly_dividend']:.2f}
Dividend Yield: {fii_data['dividend_yield']:.2f}%

---
Gerado por FII Analyzer em {datetime.now().strftime('%d/%m/%Y %H:%M')}
"""

            st.download_button(
                label="📥 Baixar Ordem (TXT)",
                data=ordem_texto,
                file_name=f"ordem_compra_{fii_data['ticker'].replace('.SA', '')}_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )


        with tab2:
            st.header("📈 Comparativo de FIIs")
            st.caption(f"Comparando {len(st.session_state.analyzed_fiis)} FII(s)")

            if len(st.session_state.analyzed_fiis) < 2:
                st.warning("⚠️ Adicione pelo menos 2 FIIs para comparar")
            else:
                comparison_data = []

                for ticker, fii_data in st.session_state.analyzed_fiis.items():
                    scores_comp = app.calculate_total_score(fii_data)
                    entry_comp = app.calculate_entry_price(fii_data)
                    rec_comp = app.get_recommendation(scores_comp['score_total'])
                    pos_comp = app.calculate_position_size(fii_data, entry_comp, 
                                                          portfolio_value, risk_tolerance)

                    comparison_data.append({
                        'Ticker': ticker.replace('.SA', ''),
                        'Score Total': scores_comp['score_total'],
                        'Score Fund.': scores_comp['score_fundamental'],
                        'Score Téc.': scores_comp['score_tecnico'],
                        'Recomendação': rec_comp['action'],
                        'Preço Atual': fii_data['preco_atual'],
                        'Entrada Sug.': entry_comp['entry_price'],
                        'Desconto %': entry_comp['discount_pct'],
                        'DY %': fii_data['dividend_yield'],
                        'P/VP': fii_data['p_vp'],
                        'Volatilidade %': fii_data['volatilidade'],
                        'Retorno 12m %': fii_data['retorno_12m'],
                        'Volume': fii_data['volume_medio'],
                        'RSI': scores_comp['detalhes_tecnico']['rsi'],
                        'Ratio R/R': pos_comp['risk_reward_ratio']
                    })

                df_comp = pd.DataFrame(comparison_data)
                df_comp = df_comp.sort_values('Score Total', ascending=False)

                st.dataframe(
                    df_comp.style.background_gradient(subset=['Score Total'], cmap='RdYlGn')
                                 .background_gradient(subset=['Score Fund.'], cmap='Blues')
                                 .background_gradient(subset=['Score Téc.'], cmap='Oranges')
                                 .format({
                                     'Preço Atual': 'R$ {:.2f}',
                                     'Entrada Sug.': 'R$ {:.2f}',
                                     'Desconto %': '{:+.1f}%',
                                     'DY %': '{:.2f}%',
                                     'P/VP': '{:.2f}',
                                     'Volatilidade %': '{:.1f}%',
                                     'Retorno 12m %': '{:+.1f}%',
                                     'Volume': '{:,.0f}',
                                     'RSI': '{:.1f}',
                                     'Ratio R/R': '{:.2f}'
                                 }),
                    use_container_width=True,
                    hide_index=True
                )

                csv_comp = df_comp.to_csv(index=False)
                st.download_button(
                    "📥 Exportar Comparativo (CSV)",
                    data=csv_comp,
                    file_name=f"comparativo_fiis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )

                st.markdown("---")
                st.subheader("📊 Gráficos Comparativos")

                col1, col2 = st.columns(2)

                with col1:
                    fig_score = px.bar(df_comp, x='Ticker', y='Score Total',
                        title='Score Total por FII', color='Score Total',
                        color_continuous_scale='RdYlGn', text='Score Total')
                    fig_score.update_traces(texttemplate='%{text:.0f}', textposition='outside')
                    fig_score.update_layout(showlegend=False, xaxis_title="", yaxis_title="Score (0-100)")
                    st.plotly_chart(fig_score, use_container_width=True)

                with col2:
                    fig_breakdown = go.Figure()
                    fig_breakdown.add_trace(go.Bar(x=df_comp['Ticker'], y=df_comp['Score Fund.'],
                        name='Fundamentalista', marker_color='#4facfe'))
                    fig_breakdown.add_trace(go.Bar(x=df_comp['Ticker'], y=df_comp['Score Téc.'],
                        name='Técnico', marker_color='#fa8231'))
                    fig_breakdown.update_layout(title='Breakdown: Fundamentalista vs Técnico',
                        barmode='stack', xaxis_title="", yaxis_title="Pontos")
                    st.plotly_chart(fig_breakdown, use_container_width=True)

                col1, col2 = st.columns(2)

                with col1:
                    fig_scatter = px.scatter(df_comp, x='DY %', y='P/VP', size='Score Total',
                        color='Score Total', hover_data=['Ticker', 'Recomendação'],
                        title='Dividend Yield vs P/VP', color_continuous_scale='RdYlGn', text='Ticker')
                    fig_scatter.add_hline(y=1.0, line_dash="dash", line_color="gray",
                                         annotation_text="P/VP = 1.0")
                    fig_scatter.update_traces(textposition='top center')
                    st.plotly_chart(fig_scatter, use_container_width=True)

                with col2:
                    fig_rsi = go.Figure()
                    colors_rsi = ['green' if x < 30 else 'red' if x > 70 else 'orange' 
                                 for x in df_comp['RSI']]
                    fig_rsi.add_trace(go.Bar(x=df_comp['Ticker'], y=df_comp['RSI'],
                        marker_color=colors_rsi, text=df_comp['RSI'].round(1), textposition='outside'))
                    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green",
                                     annotation_text="Sobrevendido")
                    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red",
                                     annotation_text="Sobrecomprado")
                    fig_rsi.update_layout(title='RSI por FII', xaxis_title="", yaxis_title="RSI",
                                         showlegend=False)
                    st.plotly_chart(fig_rsi, use_container_width=True)

                col1, col2 = st.columns(2)

                with col1:
                    fig_vol = px.bar(df_comp.sort_values('Volatilidade %'), x='Ticker', 
                        y='Volatilidade %', title='Volatilidade Anual', color='Volatilidade %',
                        color_continuous_scale='RdYlGn_r')
                    st.plotly_chart(fig_vol, use_container_width=True)

                with col2:
                    fig_ret = px.bar(df_comp.sort_values('Retorno 12m %', ascending=False),
                        x='Ticker', y='Retorno 12m %', title='Retorno Acumulado 12 Meses',
                        color='Retorno 12m %', color_continuous_scale='RdYlGn')
                    fig_ret.add_hline(y=0, line_dash="dash", line_color="gray")
                    st.plotly_chart(fig_ret, use_container_width=True)
        
        with tab3:
            st.header("💼 Carteira Sugerida")
            
            portfolio_items = []
            
            for ticker, fii_data in st.session_state.analyzed_fiis.items():
                entry_cart = app.calculate_entry_price(fii_data)
                scores_cart = app.calculate_total_score(fii_data)
                rec_cart = app.get_recommendation(scores_cart['score_total'])
                
                if scores_cart['score_total'] >= 50:
                    pos_cart = app.calculate_position_size(fii_data, entry_cart, 
                                                           portfolio_value, risk_tolerance)
                    
                    if pos_cart['quantity'] > 0:
                        portfolio_items.append({
                            'ticker': ticker,
                            'ticker_display': ticker.replace('.SA', ''),
                            'nome': fii_data['nome'],
                            'score_total': scores_cart['score_total'],
                            'score_fundamental': scores_cart['score_fundamental'],
                            'score_tecnico': scores_cart['score_tecnico'],
                            'recommendation': rec_cart['action'],
                            'entry_price': entry_cart['entry_price'],
                            'quantity': pos_cart['quantity'],
                            'investment': pos_cart['total_investment'],
                            'portfolio_pct': pos_cart['portfolio_pct'],
                            'monthly_dividend': pos_cart['monthly_dividend'],
                            'annual_dividend': pos_cart['annual_dividend'],
                            'dy': fii_data['dividend_yield']
                        })
            
            portfolio_items.sort(key=lambda x: x['score_total'], reverse=True)
            portfolio_items = portfolio_items[:10]
            
            if portfolio_items:
                total_investment = sum(item['investment'] for item in portfolio_items)
                total_monthly_dividend = sum(item['monthly_dividend'] for item in portfolio_items)
                total_annual_dividend = total_monthly_dividend * 12
                avg_score = sum(item['score_total'] for item in portfolio_items) / len(portfolio_items)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("💰 Investimento Total", f"R$ {total_investment:,.2f}",
                             f"{(total_investment/portfolio_value)*100:.1f}% do portfólio")
                with col2:
                    st.metric("💸 Dividendo Mensal", f"R$ {total_monthly_dividend:,.2f}",
                             f"{(total_monthly_dividend/total_investment)*100:.2f}% yield mensal")
                with col3:
                    annual_yield = (total_annual_dividend / total_investment) * 100 if total_investment > 0 else 0
                    st.metric("📈 Yield Anual Est.", f"{annual_yield:.2f}%",
                             f"R$ {total_annual_dividend:,.2f}/ano")
                with col4:
                    st.metric("⭐ Score Médio", f"{avg_score:.1f}/100",
                             f"{len(portfolio_items)} FII(s)")
                
                st.markdown("---")
                st.subheader("📋 Composição da Carteira")
                
                df_portfolio = pd.DataFrame([{
                    'Ticker': item['ticker_display'],
                    'Recomendação': item['recommendation'],
                    'Score': item['score_total'],
                    'Fund.': item['score_fundamental'],
                    'Téc.': item['score_tecnico'],
                    'Cotas': item['quantity'],
                    'Preço Entrada': item['entry_price'],
                    'Investimento': item['investment'],
                    'Peso %': item['portfolio_pct'],
                    'DY %': item['dy'],
                    'Div. Mensal': item['monthly_dividend']
                } for item in portfolio_items])
                
                st.dataframe(df_portfolio.style.background_gradient(subset=['Score'], cmap='RdYlGn')
                            .format({'Preço Entrada': 'R$ {:.2f}', 'Investimento': 'R$ {:.2f}',
                                    'Peso %': '{:.1f}%', 'DY %': '{:.2f}%', 'Div. Mensal': 'R$ {:.2f}'}),
                            use_container_width=True, hide_index=True)
                
                csv_portfolio = df_portfolio.to_csv(index=False)
                st.download_button("📥 Exportar Carteira (CSV)", data=csv_portfolio,
                    file_name=f"carteira_fiis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv")
                
                st.markdown("---")
                
                col1, col2 = st.columns(2)
                with col1:
                    fig_pie = px.pie(df_portfolio, values='Investimento', names='Ticker',
                        title='Distribuição da Carteira', hole=0.4)
                    st.plotly_chart(fig_pie, use_container_width=True)
                with col2:
                    fig_div = px.bar(df_portfolio, x='Ticker', y='Div. Mensal',
                        title='Dividendo Mensal por FII', color='Div. Mensal',
                        color_continuous_scale='Greens')
                    st.plotly_chart(fig_div, use_container_width=True)
            else:
                st.warning("⚠️ Nenhum FII com score ≥ 50 para compor a carteira.")

# RODAPÉ
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>✅ v4.1 FINAL - Score Completo:</strong> 60% Fundamentalista + 40% Técnico</p>
    <p>Alinhado com FII_Analyzer.py | RSI, MACD, SMA | Comparativo | Carteira</p>
</div>
""", unsafe_allow_html=True)
