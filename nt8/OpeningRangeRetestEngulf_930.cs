#region Using declarations
using System;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using NinjaTrader.Data;
using NinjaTrader.Gui;
using NinjaTrader.Gui.Tools;
using NinjaTrader.Gui.Chart;
using System.Windows.Media;

using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.Strategies;
using NinjaTrader.NinjaScript.DrawingTools;
using NinjaTrader.Cbi;
#endregion

namespace NinjaTrader.NinjaScript.Strategies
{
    public class OpeningRangeRetestEngulf_930 : Strategy
    {
        // === PARAMETERS ===
		
		[NinjaScriptProperty]
		[Display(Name = "RetestBufferTicks", GroupName = "Parameters", Order = 8)]
		public int RetestBufferTicks { get; set; } = 2;

        [NinjaScriptProperty]
        [Display(Name = "RewardRiskRatio", GroupName = "Parameters", Order = 0)]
        public double RewardRiskRatio { get; set; } = 2.0;

        [NinjaScriptProperty]
        [Display(Name = "MaxTradesPerDay", GroupName = "Parameters", Order = 1)]
        public int MaxTradesPerDay { get; set; } = 2;

        [NinjaScriptProperty]
        [Display(Name = "MaxBarsAfterBreakout", GroupName = "Parameters", Order = 2)]
        public int MaxBarsAfterBreakout { get; set; } = 20;

        [NinjaScriptProperty]
        [Display(Name = "UseEngulfOnly", GroupName = "Parameters", Order = 3)]
        public bool UseEngulfOnly { get; set; } = false;

        [NinjaScriptProperty]
        [Display(Name = "RejectionFactor", GroupName = "Parameters", Order = 4)]
        public double RejectionFactor { get; set; } = 1.5;

        [NinjaScriptProperty]
        [Display(Name = "OpeningRangeMinutes", GroupName = "Parameters", Order = 5)]
        public int OpeningRangeMinutes { get; set; } = 5;

        [NinjaScriptProperty]
        [Display(Name = "StopTradingMinutesAfterOpen", GroupName = "Parameters", Order = 6)]
        public int StopTradingMinutesAfterOpen { get; set; } = 180;   // e.g. first 3 hours

        [NinjaScriptProperty]
        [Display(Name = "Debug", GroupName = "Parameters", Order = 7)]
        public bool Debug { get; set; } = true;

        // === INTERNAL STATE ===
        private double openingRangeHigh;
        private double openingRangeLow;
        private bool   rangeComplete;

        private int tradesToday;

        private DateTime sessionOpenTime = Core.Globals.MinDate;
        private DateTime orEndTime       = Core.Globals.MinDate;
        private DateTime tradeEndTime    = Core.Globals.MinDate;

        private int barsSinceHighBreak;
        private int barsSinceLowBreak;

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Name = "OpeningRangeRetestEngulf_930";
                Calculate = Calculate.OnBarClose;
                EntriesPerDirection = 1;
                EntryHandling = EntryHandling.AllEntries;
                IsExitOnSessionCloseStrategy = true;
                ExitOnSessionCloseSeconds = 30;
                IsInstantiatedOnEachOptimizationIteration = false;

                RewardRiskRatio = 2.0;
                MaxTradesPerDay = 2;
                MaxBarsAfterBreakout = 20;
                UseEngulfOnly = false;
                RejectionFactor = 1.5;
                OpeningRangeMinutes = 5;
                StopTradingMinutesAfterOpen = 180;
                Debug = true;
            }
        }

        private void ResetSession()
        {
            openingRangeHigh = double.MinValue;
            openingRangeLow  = double.MaxValue;
            rangeComplete    = false;

            tradesToday      = 0;

            barsSinceHighBreak = int.MaxValue;
            barsSinceLowBreak  = int.MaxValue;

            // clear old OR lines
            string tagHigh = "ORHigh_" + sessionOpenTime.ToString("yyyyMMdd_HHmm");
            string tagLow  = "ORLow_"  + sessionOpenTime.ToString("yyyyMMdd_HHmm");
            RemoveDrawObject(tagHigh);
            RemoveDrawObject(tagLow);
        }

        protected override void OnBarUpdate()
        {
            if (CurrentBar < BarsRequiredToTrade)
                return;

            DateTime bt = Times[0][0];

            // === 1) Detect new RTH session ===
            if (Bars.IsFirstBarOfSession)
            {
                sessionOpenTime = bt;
                orEndTime       = sessionOpenTime.AddMinutes(OpeningRangeMinutes);
                tradeEndTime    = (StopTradingMinutesAfterOpen > 0)
                                  ? sessionOpenTime.AddMinutes(StopTradingMinutesAfterOpen)
                                  : DateTime.MaxValue;

                ResetSession();
                D(string.Format("=== NEW SESSION {0} ===", sessionOpenTime));
            }

            if (sessionOpenTime == Core.Globals.MinDate)
                return; // safety

            // === 2) Build opening range from session open to session open + OR minutes ===
            if (!rangeComplete)
            {
                if (bt <= orEndTime)
                {
                    openingRangeHigh = Math.Max(openingRangeHigh, High[0]);
                    openingRangeLow  = Math.Min(openingRangeLow, Low[0]);
                    return; // still building range
                }
                else if (openingRangeHigh > double.MinValue && openingRangeLow < double.MaxValue)
                {
                    rangeComplete = true;
                    D(string.Format("{0} OR DONE  High={1}  Low={2}", bt, openingRangeHigh, openingRangeLow));

                    string tagHigh = "ORHigh_" + sessionOpenTime.ToString("yyyyMMdd_HHmm");
                    string tagLow  = "ORLow_"  + sessionOpenTime.ToString("yyyyMMdd_HHmm");
                    Draw.HorizontalLine(this, tagHigh, openingRangeHigh, Brushes.DodgerBlue);
                    Draw.HorizontalLine(this, tagLow,  openingRangeLow,  Brushes.OrangeRed);
                }
            }

            if (!rangeComplete)
                return;

            if (bt > tradeEndTime)
                return;

            if (tradesToday >= MaxTradesPerDay)
                return;

            // === 3) Track breakout state ===
            if (barsSinceHighBreak != int.MaxValue)
                barsSinceHighBreak++;
            if (barsSinceLowBreak != int.MaxValue)
                barsSinceLowBreak++;

            if (Close[0] > openingRangeHigh)
            {
                barsSinceHighBreak = 0;
                D(string.Format("{0} BREAKOUT HIGH  Close={1} ORHigh={2}", bt, Close[0], openingRangeHigh));
            }

            if (Close[0] < openingRangeLow)
            {
                barsSinceLowBreak = 0;
                D(string.Format("{0} BREAKOUT LOW   Close={1} ORLow={2}", bt, Close[0], openingRangeLow));
            }

            // Only when flat
            if (Position.MarketPosition != MarketPosition.Flat)
                return;

            // === 4) LONG SETUP ===
            bool longSignal = false;
			double buffer = RetestBufferTicks * TickSize;
			
            if (barsSinceHighBreak > 0 && barsSinceHighBreak <= MaxBarsAfterBreakout)
            {
                bool touchedLevel  = Low[0] <= openingRangeHigh + buffer 
                     && High[0] >= openingRangeHigh - buffer 
                     && Close[0] > openingRangeHigh;
				
                bool bullishEngulf = IsBullishEngulfing(0);
                bool bullishReject = IsBullishRejection(0);

                if (touchedLevel)
                    D(string.Format("{0} LONG touch  Close={1} Low={2} ORHigh={3}", bt, Close[0], Low[0], openingRangeHigh));
                if (bullishEngulf)
                    D(string.Format("{0} LONG engulf candidate", bt));
                if (bullishReject)
                    D(string.Format("{0} LONG rejection candidate", bt));

                if (touchedLevel && (bullishEngulf || (!UseEngulfOnly && bullishReject)))
                    longSignal = true;
            }

            // === 5) SHORT SETUP ===
            bool shortSignal = false;
            if (barsSinceLowBreak > 0 && barsSinceLowBreak <= MaxBarsAfterBreakout)
            {
                
				bool touchedLevel   = High[0] >= openingRangeLow - buffer
                      && Low[0]  <= openingRangeLow + buffer
                      && Close[0] < openingRangeLow;

                bool bearishEngulf  = IsBearishEngulfing(0);
                bool bearishReject  = IsBearishRejection(0);

                if (touchedLevel)
                    D(string.Format("{0} SHORT touch  Close={1} High={2} ORLow={3}", bt, Close[0], High[0], openingRangeLow));
                if (bearishEngulf)
                    D(string.Format("{0} SHORT engulf candidate", bt));
                if (bearishReject)
                    D(string.Format("{0} SHORT rejection candidate", bt));

                if (touchedLevel && (bearishEngulf || (!UseEngulfOnly && bearishReject)))
                    shortSignal = true;
            }

            // === 6) EXECUTE ENTRIES ===
            if (longSignal)
            {
                double entryPrice = Close[0];
                double stopPrice  = Low[0];
                double risk       = entryPrice - stopPrice;
                if (risk <= TickSize * 2)
                    return;

                double targetPrice = entryPrice + risk * RewardRiskRatio;

                SetStopLoss(CalculationMode.Price, stopPrice);
                SetProfitTarget(CalculationMode.Price, targetPrice);

                D(string.Format("{0} ENTER LONG  entry={1} stop={2} target={3}", bt, entryPrice, stopPrice, targetPrice));
                EnterLong();
                tradesToday++;
                return;
            }

            if (shortSignal)
            {
                double entryPrice = Close[0];
                double stopPrice  = High[0];
                double risk       = stopPrice - entryPrice;
                if (risk <= TickSize * 2)
                    return;

                double targetPrice = entryPrice - risk * RewardRiskRatio;

                SetStopLoss(CalculationMode.Price, stopPrice);
                SetProfitTarget(CalculationMode.Price, targetPrice);

                D(string.Format("{0} ENTER SHORT entry={1} stop={2} target={3}", bt, entryPrice, stopPrice, targetPrice));
                EnterShort();
                tradesToday++;
                return;
            }
        }

        // === PATTERN HELPERS ===
        private bool IsBullishEngulfing(int idx)
        {
            if (CurrentBar < idx + 2)
                return false;

            if (Close[idx + 1] >= Open[idx + 1]) // previous must be bearish
                return false;

            bool currentBullish = Close[idx] > Open[idx];
            bool bodyEngulfs =
                Close[idx] >= Open[idx + 1] &&
                Open[idx]  <= Close[idx + 1];

            return currentBullish && bodyEngulfs;
        }

        private bool IsBearishEngulfing(int idx)
        {
            if (CurrentBar < idx + 2)
                return false;

            if (Close[idx + 1] <= Open[idx + 1]) // previous must be bullish
                return false;

            bool currentBearish = Close[idx] < Open[idx];
            bool bodyEngulfs =
                Close[idx] <= Open[idx + 1] &&
                Open[idx]  >= Close[idx + 1];

            return currentBearish && bodyEngulfs;
        }

        private bool IsBullishRejection(int idx)
        {
            double body = Math.Abs(Close[idx] - Open[idx]);
            if (body <= 0)
                return false;

            double lowerWick = Math.Min(Open[idx], Close[idx]) - Low[idx];

            return Close[idx] > Open[idx] && lowerWick > body * RejectionFactor;
        }

        private bool IsBearishRejection(int idx)
        {
            double body = Math.Abs(Close[idx] - Open[idx]);
            if (body <= 0)
                return false;

            double upperWick = High[idx] - Math.Max(Open[idx], Close[idx]);

            return Close[idx] < Open[idx] && upperWick > body * RejectionFactor;
        }

        // === DEBUG PRINT ===
        private void D(string msg)
        {
            if (Debug)
                Print(msg);
        }
    }
}
