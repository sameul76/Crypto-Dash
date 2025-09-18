# Replace the chart creation section in your tab1 with this improved version:

with tab1:
    assets = sorted(market_df["asset"].dropna().unique()) if (not market_df.empty and "asset" in market_df.columns) else []
    if assets:
        default_index = assets.index(DEFAULT_ASSET) if DEFAULT_ASSET in assets else 0
        selected_asset = st.selectbox("Select Asset to View", assets, index=default_index, key="asset_select")
        range_choice = st.selectbox("Select Date Range", ["30 days", "7 days", "1 day", "All"], index=0, key="range_select")

        # Last price metric
        asset_market_data = market_df[market_df['asset'] == selected_asset] if not market_df.empty else pd.DataFrame()
        if not asset_market_data.empty:
            last_price = asset_market_data.sort_values('timestamp').iloc[-1]['close']
            st.metric(f"Last Price for {selected_asset}", f"${last_price:,.6f}" if last_price < 1 else f"${last_price:,.2f}")

        st.markdown("---")

        df = asset_market_data.sort_values("timestamp")
        if not df.empty:
            end_date = df["timestamp"].max()
            if range_choice == "1 day":
                start_date = end_date - timedelta(days=1)
            elif range_choice == "7 days":
                start_date = end_date - timedelta(days=7)
            elif range_choice == "30 days":
                start_date = end_date - timedelta(days=30)
            else:
                start_date = df["timestamp"].min()

            vis = df[(df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)].copy()

            if not vis.empty and len(vis) > 1:
                # Create the main candlestick chart
                fig = go.Figure()
                
                # Add candlestick data
                fig.add_trace(go.Candlestick(
                    x=vis["timestamp"],
                    open=vis["open"],
                    high=vis["high"], 
                    low=vis["low"],
                    close=vis["close"],
                    name=selected_asset,
                    increasing_line_color='#00ff88',
                    decreasing_line_color='#ff4444',
                    increasing_fillcolor='rgba(0, 255, 136, 0.3)',
                    decreasing_fillcolor='rgba(255, 68, 68, 0.3)'
                ))

                # Add ML probability data as a separate subplot or overlay
                if 'p_up' in vis.columns and 'p_down' in vis.columns:
                    # Create hover text for probabilities
                    hover_text = []
                    for _, row in vis.iterrows():
                        p_up = row.get('p_up', float('nan'))
                        p_down = row.get('p_down', float('nan'))
                        if pd.notna(p_up) and pd.notna(p_down):
                            text = f"P(Up): {p_up:.3f} | P(Down): {p_down:.3f}"
                            hover_text.append(text)
                        else:
                            hover_text.append("")
                    
                    # Add invisible scatter trace for probability hover info
                    fig.add_trace(go.Scatter(
                        x=vis["timestamp"],
                        y=vis["close"],
                        mode='markers',
                        marker=dict(size=8, opacity=0),  # Invisible markers
                        text=hover_text,
                        hovertemplate='<b>%{text}</b><br>Close: $%{y:.6f}<extra></extra>',
                        name='ML Probabilities',
                        showlegend=False
                    ))

                # Add trade markers
                if not trades_df.empty:
                    asset_trades = trades_df[
                        (trades_df["asset"] == selected_asset) &
                        (trades_df["timestamp"] >= start_date) &
                        (trades_df["timestamp"] <= end_date)
                    ].copy()
                    
                    if not asset_trades.empty:
                        # Add buy trades
                        buy_trades = asset_trades[asset_trades["action"] == "buy"]
                        if not buy_trades.empty:
                            fig.add_trace(go.Scatter(
                                x=buy_trades["timestamp"], 
                                y=buy_trades["price"],
                                mode="markers+text",
                                name="BUY",
                                marker=dict(
                                    symbol='triangle-up',
                                    size=15,
                                    color='#00ff88',
                                    line=dict(width=2, color='#ffffff')
                                ),
                                text=['BUY'] * len(buy_trades),
                                textposition="top center",
                                textfont=dict(size=10, color='#00ff88'),
                                hovertemplate='<b>BUY</b><br>Price: $%{y:.6f}<br>Reason: %{customdata}<extra></extra>',
                                customdata=buy_trades.get('reason', '')
                            ))
                        
                        # Add sell trades
                        sell_trades = asset_trades[asset_trades["action"] == "sell"]
                        if not sell_trades.empty:
                            fig.add_trace(go.Scatter(
                                x=sell_trades["timestamp"],
                                y=sell_trades["price"], 
                                mode="markers+text",
                                name="SELL",
                                marker=dict(
                                    symbol='triangle-down',
                                    size=15,
                                    color='#ff4444',
                                    line=dict(width=2, color='#ffffff')
                                ),
                                text=['SELL'] * len(sell_trades),
                                textposition="bottom center", 
                                textfont=dict(size=10, color='#ff4444'),
                                hovertemplate='<b>SELL</b><br>Price: $%{y:.6f}<br>Reason: %{customdata}<extra></extra>',
                                customdata=sell_trades.get('reason', '')
                            ))

                        # Add P&L lines between buy/sell pairs
                        sorted_trades = asset_trades.sort_values("timestamp")
                        open_trades = []
                        
                        for _, trade in sorted_trades.iterrows():
                            if trade.get('action') == 'buy':
                                open_trades.append(trade)
                            elif trade.get('action') == 'sell' and open_trades:
                                # Match with oldest buy (FIFO)
                                buy_trade = open_trades.pop(0)
                                pnl = float(trade.get('price', 0)) - float(buy_trade.get('price', 0))
                                pnl_pct = (pnl / float(buy_trade.get('price', 1))) * 100
                                
                                line_color = "#00ff88" if pnl >= 0 else "#ff4444"
                                line_width = 3 if abs(pnl_pct) > 5 else 2  # Thicker lines for bigger moves
                                
                                fig.add_trace(go.Scatter(
                                    x=[buy_trade['timestamp'], trade['timestamp']],
                                    y=[buy_trade['price'], trade['price']],
                                    mode='lines',
                                    line=dict(color=line_color, width=line_width, dash='solid'),
                                    showlegend=False,
                                    hovertemplate=f'P&L: ${pnl:.4f} ({pnl_pct:+.2f}%)<extra></extra>',
                                    name=f'Trade P&L'
                                ))

                # Update layout for better visualization
                price_range = vis['high'].max() - vis['low'].min()
                y_padding = price_range * 0.1  # 10% padding
                
                fig.update_layout(
                    title=f"{selected_asset} — Price & Trade Activity ({range_choice})",
                    template="plotly_white",
                    xaxis_rangeslider_visible=False,
                    xaxis=dict(
                        title="Date/Time",
                        type='date',
                        tickformat='%m/%d %H:%M' if range_choice in ["1 day", "7 days"] else '%m/%d',
                        showgrid=True,
                        gridcolor='rgba(128,128,128,0.2)'
                    ),
                    yaxis=dict(
                        title="Price (USD)",
                        tickformat='.6f' if vis['close'].iloc[-1] < 1 else '.2f',
                        showgrid=True,
                        gridcolor='rgba(128,128,128,0.2)',
                        range=[vis['low'].min() - y_padding, vis['high'].max() + y_padding]
                    ),
                    hovermode="x unified",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="center",
                        x=0.5
                    ),
                    height=700,
                    margin=dict(l=60, r=20, t=80, b=60),
                    plot_bgcolor='rgba(248,248,248,0.8)'
                )
                
                # Add volume bars if available
                if 'volume' in vis.columns and vis['volume'].notna().any():
                    # Create secondary y-axis for volume
                    fig.update_layout(yaxis2=dict(
                        title="Volume",
                        overlaying='y',
                        side='right',
                        showgrid=False
                    ))
                    
                    fig.add_trace(go.Bar(
                        x=vis["timestamp"],
                        y=vis["volume"],
                        name="Volume",
                        yaxis='y2',
                        opacity=0.3,
                        marker_color='blue'
                    ))

                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
                
                # Add summary stats for the selected period
                if not asset_trades.empty:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Trades in Period", len(asset_trades))
                    with col2:
                        buy_count = len(asset_trades[asset_trades['action'] == 'buy'])
                        st.metric("Buy Orders", buy_count)
                    with col3:
                        sell_count = len(asset_trades[asset_trades['action'] == 'sell'])
                        st.metric("Sell Orders", sell_count)
                    with col4:
                        if 'pnl' in asset_trades.columns:
                            period_pnl = asset_trades['pnl'].sum()
                            st.metric("Period P&L", f"${period_pnl:.4f}")
                        
            elif len(vis) <= 1:
                st.warning(f"Insufficient data points ({len(vis)}) for {selected_asset} in the selected date range.")
            else:
                st.warning(f"No data for {selected_asset} in the selected date range.")
        else:
            st.warning(f"No market data found for {selected_asset}.")
    else:
        st.warning("Market data not loaded or available.")
