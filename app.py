import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from fredapi import Fred
from groq import Groq
from dotenv import load_dotenv
import os
import math

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
FRED_API_KEY = os.getenv("FRED_API_KEY")

st.set_page_config(page_title="Dashboard Financiero ML", layout="wide")
st.title("📊 Dashboard Financiero ML")
st.caption("Plataforma de inteligencia artificial para análisis financiero y valoración de empresas")

ticker = st.text_input("Ingresa el Ticker del NYSE", "AAPL").upper().strip()

if ticker:
    with st.spinner("Cargando datos..."):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            if not info.get("currentPrice") and not info.get("regularMarketPrice"):
                st.error("Ticker no encontrado. Ingresa un ticker válido del NYSE.")
                st.stop()

            price = info.get("currentPrice") or info.get("regularMarketPrice")
            beta = info.get("beta")
            market_cap = info.get("marketCap")
            sector = info.get("sector", "N/A")
            name = info.get("longName", ticker)

            st.subheader(f"{name} — {sector}")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Precio Actual", f"${price:,.2f}" if price else "N/A")
            col2.metric("Beta", f"{beta:.2f}" if beta else "N/A")
            col3.metric("Capitalización", f"${market_cap/1e9:.1f}B" if market_cap else "N/A")
            col4.metric("Sector", sector)

            with st.expander("Descripción de la Empresa"):
                st.write(info.get("longBusinessSummary", "No disponible."))

            # --- GRAFICO PRECIO ---
            st.subheader("📈 Historial del Precio")
            period = st.selectbox("Período", ["6mo", "1y", "2y", "5y"], index=1)
            hist = stock.history(period=period)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist.index, y=hist["Close"],
                mode='lines', name='Precio de Cierre',
                line=dict(color='#00D4FF', width=2)
            ))
            fig.update_layout(template="plotly_dark", height=400,
                            xaxis_title="Fecha", yaxis_title="Precio (USD)")
            st.plotly_chart(fig, use_container_width=True)

            # --- TASA LIBRE DE RIESGO ---
            fred = Fred(api_key=FRED_API_KEY)
            treasury = fred.get_series('GS10').dropna()
            rf = float(treasury.iloc[-1]) / 100

            # --- PRIMA DE RIESGO HISTORICA ---
            sp500 = yf.Ticker("^GSPC")
            sp500_hist = sp500.history(period="5y")["Close"]
            sp500_return = float(sp500_hist.pct_change().dropna().mean() * 252)
            treasury_5y = fred.get_series('GS10', observation_start='2020-01-01').dropna()
            rf_avg = float(treasury_5y.mean()) / 100
            market_premium = sp500_return - rf_avg

            # --- CAPM ---
            beta_val = beta if beta else 1.0
            ke = rf + beta_val * market_premium

            # --- ESTADOS FINANCIEROS ---
            financials = stock.financials
            balance = stock.balance_sheet
            cashflow = stock.cashflow

            interest_expense = 0
            total_debt = 0
            cash = 0
            tax_rate = 0.21

            try:
                interest_expense = abs(float(financials.loc["Interest Expense"].iloc[0]))
            except:
                pass
            try:
                total_debt = float(balance.loc["Total Debt"].iloc[0])
            except:
                pass
            try:
                cash = float(balance.loc["Cash And Cash Equivalents"].iloc[0])
            except:
                try:
                    cash = float(balance.loc["Cash Cash Equivalents And Short Term Investments"].iloc[0])
                except:
                    cash = 0
            try:
                tax_provision = float(financials.loc["Tax Provision"].iloc[0])
                pretax_income = float(financials.loc["Pretax Income"].iloc[0])
                if pretax_income > 0:
                    tax_rate = tax_provision / pretax_income
                    tax_rate = min(max(tax_rate, 0.15), 0.35)
            except:
                pass

            kd = (interest_expense / total_debt) if total_debt > 0 else 0.05
            kd = min(max(kd, 0.02), 0.10)
            kd_after_tax = kd * (1 - tax_rate)

            equity = float(market_cap) if market_cap else 0
            debt = float(total_debt) if total_debt else 0
            total_capital = equity + debt

            if total_capital > 0 and equity > 0:
                wacc = (equity/total_capital) * ke + (debt/total_capital) * kd_after_tax
            else:
                wacc = ke

            if math.isnan(wacc) or wacc <= 0:
                wacc = 0.10

            # --- DCF ---
            try:
                op_cf = cashflow.loc["Operating Cash Flow"].dropna()
                capex = cashflow.loc["Capital Expenditure"].dropna()
                fcf_row = (op_cf + capex).dropna()
            except:
                fcf_row = pd.Series([])

            fcf_values = fcf_row.values[::-1]
            fcf_positivos = [float(v) for v in fcf_values if v > 0]

            if len(fcf_positivos) >= 3:
                base_fcf = float(pd.Series(fcf_positivos[-4:]).mean())
            elif len(fcf_positivos) >= 1:
                base_fcf = float(pd.Series(fcf_positivos).mean())
            else:
                try:
                    base_fcf = float(financials.loc["EBITDA"].iloc[0]) * 0.6
                except:
                    base_fcf = float(market_cap) * 0.04 if market_cap else 1e9

            if len(fcf_positivos) >= 2:
                growth_rates = []
                for i in range(1, len(fcf_positivos)):
                    g = (fcf_positivos[i] - fcf_positivos[i-1]) / abs(fcf_positivos[i-1])
                    growth_rates.append(g)
                growth_rate = float(pd.Series(growth_rates).mean())
                growth_rate = min(max(growth_rate, 0.03), 0.12)
            else:
                growth_rate = 0.05

            terminal_growth = 0.03

            if wacc <= terminal_growth:
                wacc = terminal_growth + 0.04

            projected_fcf = [base_fcf * (1 + growth_rate) ** i for i in range(1, 6)]
            discount_factors = [(1 / (1 + wacc) ** i) for i in range(1, 6)]
            pv_fcf = sum([f * d for f, d in zip(projected_fcf, discount_factors)])

            terminal_value = (projected_fcf[-1] * (1 + terminal_growth)) / (wacc - terminal_growth)
            pv_terminal = terminal_value / (1 + wacc) ** 5

            enterprise_value = pv_fcf + pv_terminal
            net_debt = max(debt - cash, 0)
            equity_value = enterprise_value - net_debt + cash

            shares = info.get("sharesOutstanding", 1)
            intrinsic_value = equity_value / shares if shares > 0 else 0

            upside = (intrinsic_value - price) / price if price else 0
            annual_return = upside / 5
            monthly_return = (1 + annual_return) ** (1/12) - 1

            # --- MOSTRAR CAPM Y WACC ---
            st.subheader("📐 Análisis CAPM & WACC")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Tasa Libre de Riesgo (Rf)", f"{rf*100:.2f}%")
            c2.metric("Prima de Riesgo del Mercado", f"{market_premium*100:.2f}%")
            c3.metric("Costo de Capital (Ke)", f"{ke*100:.2f}%")
            c4.metric("WACC", f"{wacc*100:.2f}%")

            st.latex(r"E(R_i) = R_f + \beta_i \cdot (E(R_m) - R_f)")

            # --- MOSTRAR DCF ---
            st.subheader("💰 Valoración por DCF")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Valor Intrínseco", f"${intrinsic_value:,.2f}")
            c2.metric("Precio Actual", f"${price:,.2f}")
            c3.metric("Potencial Upside/Downside", f"{upside*100:.1f}%")
            c4.metric("Retorno Anual Esperado", f"{annual_return*100:.2f}%")

            st.info(
                f"📊 **Supuestos del modelo:** "
                f"FCF base promedio: ${base_fcf/1e9:.2f}B | "
                f"Tasa crecimiento: {growth_rate*100:.2f}% | "
                f"Crecimiento perpetuo: {terminal_growth*100:.1f}% | "
                f"Retorno mensual esperado: {monthly_return*100:.2f}%"
            )

            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=[f"Año {i+1}" for i in range(5)],
                y=[p/1e9 for p in projected_fcf],
                name="FCF Proyectado",
                marker_color='#00D4FF'
            ))
            fig2.update_layout(
                template="plotly_dark",
                title="Flujos de Caja Libres Proyectados (5 Años)",
                yaxis_title="Miles de Millones USD",
                height=350
            )
            st.plotly_chart(fig2, use_container_width=True)

            with st.expander("🔍 Ver desglose detallado del DCF"):
                st.write(f"**Valor Presente de Flujos (5 años):** ${pv_fcf/1e9:,.2f}B")
                st.write(f"**Valor Terminal Descontado:** ${pv_terminal/1e9:,.2f}B")
                st.write(f"**Valor Empresa (EV):** ${enterprise_value/1e9:,.2f}B")
                st.write(f"**(-) Deuda Neta:** ${net_debt/1e9:,.2f}B")
                st.write(f"**(+) Efectivo:** ${cash/1e9:,.2f}B")
                st.write(f"**Valor Patrimonial (Equity):** ${equity_value/1e9:,.2f}B")
                st.write(f"**Acciones en Circulación:** {shares/1e9:,.2f}B")
                st.write(f"**Valor Intrínseco por Acción:** ${intrinsic_value:,.2f}")

            # =============================================
            # 🧠 ANÁLISIS DE SENTIMIENTO DE NOTICIAS (ML)
            # =============================================
            st.subheader("🧠 Análisis de Sentimiento de Noticias (ML)")

            news = stock.news
            if news and len(news) > 0:
                try:
                    client = Groq(api_key=GROQ_API_KEY)
                    
                    # Preparar titulares
                    headlines = []
                    for n in news[:8]:
                        title = n.get("title", "")
                        if title:
                            headlines.append(title)

                    if headlines:
                        headlines_text = "\n".join([f"{i+1}. {h}" for i, h in enumerate(headlines)])

                        sentiment_response = client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[
                                {"role": "system", "content": """Eres un modelo de ML de análisis de sentimiento financiero.
Para cada titular responde EXACTAMENTE en este formato (una línea por titular):
NUMERO|SENTIMIENTO|PUNTAJE
Donde SENTIMIENTO es: POSITIVO, NEGATIVO o NEUTRAL
Y PUNTAJE es un número entre -100 y 100.
No escribas nada más, solo las líneas con el formato indicado."""},
                                {"role": "user", "content": f"Clasifica el sentimiento de estos titulares financieros sobre {name}:\n{headlines_text}"}
                            ]
                        )

                        sentiment_text = sentiment_response.choices[0].message.content.strip()
                        
                        positivos = 0
                        negativos = 0
                        neutrales = 0
                        scores = []
                        resultados = []

                        for line in sentiment_text.split("\n"):
                            parts = line.strip().split("|")
                            if len(parts) >= 3:
                                try:
                                    idx = int(parts[0].strip()) - 1
                                    sent = parts[1].strip().upper()
                                    score = int(parts[2].strip())
                                    scores.append(score)

                                    if "POSITIVO" in sent:
                                        positivos += 1
                                        emoji = "🟢"
                                    elif "NEGATIVO" in sent:
                                        negativos += 1
                                        emoji = "🔴"
                                    else:
                                        neutrales += 1
                                        emoji = "🟡"

                                    if idx < len(headlines):
                                        resultados.append({
                                            "Noticia": headlines[idx][:80],
                                            "Sentimiento": f"{emoji} {sent}",
                                            "Puntaje": score
                                        })
                                except:
                                    pass

                        if resultados:
                            avg_score = sum(scores) / len(scores) if scores else 0

                            # Indicador general
                            if avg_score > 20:
                                sentimiento_general = "🟢 POSITIVO"
                                color = "green"
                            elif avg_score < -20:
                                sentimiento_general = "🔴 NEGATIVO"
                                color = "red"
                            else:
                                sentimiento_general = "🟡 NEUTRAL"
                                color = "orange"

                            mc1, mc2, mc3, mc4 = st.columns(4)
                            mc1.metric("Sentimiento General", sentimiento_general)
                            mc2.metric("Puntaje Promedio", f"{avg_score:.0f}/100")
                            mc3.metric("Noticias Positivas", f"{positivos}")
                            mc4.metric("Noticias Negativas", f"{negativos}")

                            # Tabla de noticias
                            df_news = pd.DataFrame(resultados)
                            st.dataframe(df_news, use_container_width=True, hide_index=True)

                            # Grafico de sentimiento
                            fig_sent = go.Figure()
                            colors = ['#00D4FF' if s > 0 else '#FF4444' if s < 0 else '#FFD700' for s in scores]
                            fig_sent.add_trace(go.Bar(
                                x=[f"N{i+1}" for i in range(len(scores))],
                                y=scores,
                                marker_color=colors
                            ))
                            fig_sent.update_layout(
                                template="plotly_dark",
                                title="Puntaje de Sentimiento por Noticia",
                                yaxis_title="Puntaje (-100 a 100)",
                                height=300
                            )
                            st.plotly_chart(fig_sent, use_container_width=True)
                        else:
                            st.warning("No se pudo procesar el sentimiento de las noticias.")
                    else:
                        st.info("No hay titulares disponibles para analizar.")
                except Exception as e:
                    st.warning(f"Error en análisis de sentimiento: {e}")
            else:
                st.info("No hay noticias recientes disponibles para esta empresa.")

            # =============================================
            # 🎯 SCORING DE INVERSIÓN (ML)
            # =============================================
            st.subheader("🎯 Scoring de Inversión (ML)")

            # Calcular score basado en múltiples factores
            score_components = {}

            # 1. Upside score (0-25 puntos)
            if upside > 0.3:
                score_components["Potencial de Valorización"] = 25
            elif upside > 0.1:
                score_components["Potencial de Valorización"] = 20
            elif upside > 0:
                score_components["Potencial de Valorización"] = 15
            elif upside > -0.2:
                score_components["Potencial de Valorización"] = 10
            else:
                score_components["Potencial de Valorización"] = 5

            # 2. Beta / Riesgo (0-20 puntos)
            if beta_val < 0.8:
                score_components["Nivel de Riesgo (Beta)"] = 20
            elif beta_val < 1.2:
                score_components["Nivel de Riesgo (Beta)"] = 15
            elif beta_val < 1.5:
                score_components["Nivel de Riesgo (Beta)"] = 10
            else:
                score_components["Nivel de Riesgo (Beta)"] = 5

            # 3. FCF positivo y creciente (0-20 puntos)
            if len(fcf_positivos) >= 4:
                score_components["Flujo de Caja Libre"] = 20
            elif len(fcf_positivos) >= 2:
                score_components["Flujo de Caja Libre"] = 15
            elif len(fcf_positivos) >= 1:
                score_components["Flujo de Caja Libre"] = 10
            else:
                score_components["Flujo de Caja Libre"] = 5

            # 4. Crecimiento (0-15 puntos)
            if growth_rate > 0.08:
                score_components["Tasa de Crecimiento"] = 15
            elif growth_rate > 0.05:
                score_components["Tasa de Crecimiento"] = 12
            elif growth_rate > 0.03:
                score_components["Tasa de Crecimiento"] = 8
            else:
                score_components["Tasa de Crecimiento"] = 5

            # 5. WACC eficiente (0-10 puntos)
            if wacc < 0.08:
                score_components["Eficiencia del Capital"] = 10
            elif wacc < 0.12:
                score_components["Eficiencia del Capital"] = 7
            else:
                score_components["Eficiencia del Capital"] = 4

            # 6. Sentimiento noticias (0-10 puntos)
            try:
                if avg_score > 20:
                    score_components["Sentimiento del Mercado"] = 10
                elif avg_score > 0:
                    score_components["Sentimiento del Mercado"] = 7
                elif avg_score > -20:
                    score_components["Sentimiento del Mercado"] = 5
                else:
                    score_components["Sentimiento del Mercado"] = 2
            except:
                score_components["Sentimiento del Mercado"] = 5

            total_score = sum(score_components.values())

            # Clasificación
            if total_score >= 80:
                clasificacion = "🟢 COMPRA FUERTE"
                desc_clasif = "La empresa muestra fundamentos sólidos y buenas perspectivas."
            elif total_score >= 65:
                clasificacion = "🟢 COMPRA"
                desc_clasif = "La empresa presenta buenas métricas generales."
            elif total_score >= 50:
                clasificacion = "🟡 MANTENER"
                desc_clasif = "La empresa muestra fundamentos mixtos. Se recomienda vigilar."
            elif total_score >= 35:
                clasificacion = "🟠 PRECAUCIÓN"
                desc_clasif = "Algunos indicadores muestran debilidad. Analizar con cuidado."
            else:
                clasificacion = "🔴 VENTA"
                desc_clasif = "Los fundamentos muestran señales de riesgo significativas."

            # Mostrar score
            sc1, sc2 = st.columns([1, 2])
            with sc1:
                st.metric("Score Total", f"{total_score}/100")
                st.metric("Clasificación", clasificacion)
                st.write(desc_clasif)

            with sc2:
                df_scores = pd.DataFrame(
                    list(score_components.items()),
                    columns=["Factor", "Puntaje"]
                )
                fig_score = go.Figure()
                fig_score.add_trace(go.Bar(
                    x=df_scores["Puntaje"],
                    y=df_scores["Factor"],
                    orientation='h',
                    marker_color=['#00D4FF' if p > 12 else '#FFD700' if p > 7 else '#FF4444' for p in df_scores["Puntaje"]]
                ))
                fig_score.update_layout(
                    template="plotly_dark",
                    title="Desglose del Score de Inversión",
                    xaxis_title="Puntaje",
                    height=350
                )
                st.plotly_chart(fig_score, use_container_width=True)

            # --- ESTADOS FINANCIEROS ---
            st.subheader("📋 Estados Financieros")
            st.dataframe(financials, use_container_width=True)

            # --- ASISTENTE IA ---
            st.subheader("🤖 Asistente Financiero IA")
            st.caption("Pregúntale al asistente sobre esta empresa")

            user_question = st.text_input(
                "Tu pregunta",
                f"Analiza {name} y explica los resultados de la valoración DCF en español"
            )

            if st.button("Analizar con IA"):
                with st.spinner("El asistente está analizando..."):
                    context = f"""
                    Empresa: {name} | Ticker: {ticker} | Sector: {sector}
                    Precio Actual: ${price:,.2f}
                    Beta: {beta_val}
                    Tasa Libre de Riesgo: {rf*100:.2f}%
                    Prima de Riesgo del Mercado: {market_premium*100:.2f}%
                    Costo de Capital (Ke): {ke*100:.2f}%
                    WACC: {wacc*100:.2f}%
                    Tasa de Crecimiento: {growth_rate*100:.2f}%
                    FCF Base Promedio: ${base_fcf/1e9:.2f}B
                    Valor Intrínseco: ${intrinsic_value:,.2f}
                    Upside/Downside: {upside*100:.1f}%
                    Score de Inversión: {total_score}/100
                    Clasificación: {clasificacion}
                    Retorno Anual Esperado: {annual_return*100:.2f}%
                    """
                    try:
                        client = Groq(api_key=GROQ_API_KEY)
                        response = client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[
                                {"role": "system", "content": "Eres un analista financiero experto en valoración de empresas, análisis DCF e inversiones. Responde siempre en español de forma clara y profesional. Explica el contexto de los números, el análisis de sentimiento, el score de inversión, y da una recomendación clara al final."},
                                {"role": "user", "content": f"Contexto financiero:\n{context}\n\nPregunta: {user_question}"}
                            ]
                        )
                        st.write(response.choices[0].message.content)
                    except Exception as e:
                        st.error(f"Error en el asistente IA: {e}")

        except Exception as e:
            st.error(f"Error cargando datos: {e}")
