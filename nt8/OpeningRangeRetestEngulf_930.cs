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
using NinjaTrader.NinjaScript.Indicators;
#endregion

namespace NinjaTrader.NinjaScript.Strategies
{
    public class OpeningRangeRetestEngulf_Permissive : Strategy
    {
        // === PARAMETERS ===

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
        [Range(5, 30)]
        public int OpeningRangeMinutes { get; set; } = 15;

        [NinjaScriptProperty]
        [Display(Name = "StopTradingMinutesAfterOpen", GroupName = "Parameters", Order = 6)]
        public int StopTradingMinutesAfterOpen { get; set; } = 180;

        [NinjaScriptProperty]
        [Display(Name = "Debug", GroupName = "Parameters", Order = 7)]
        public bool Debug { get; set; } = true;

        [NinjaScriptProperty]
        [Display(Name = "RetestBufferTicks", GroupName = "Parameters", Order = 8)]
        public int RetestBufferTicks { get; set; } = 3;

        [NinjaScriptProperty]
        [Display(Name = "RequireRetest", GroupName = "Parameters", Order = 9)]
        public bool RequireRetest { get; set; } = false;

        [NinjaScriptProperty]
        [Display(Name = "MaxDistanceFromOR", GroupName = "Parameters", Order = 10)]
        public int MaxDistanceFromOR { get; set; } = 20;   // ticks

        [NinjaScriptProperty]
        [Display(Name = "EnableLongs", GroupName = "Parameters", Order = 11)]
        public bool EnableLongs { get; set; } = true;

        [NinjaScriptProperty]
        [Display(Name = "EnableShorts", GroupName = "Parameters", Order = 12)]
        public bool EnableShorts { get; set; } = true;

        [NinjaScriptProperty]
        [Display(Name = "MinORTicks", GroupName = "Parameters", Order = 13)]
        public int MinORTicks { get; set; } = 10;

        [NinjaScriptProperty]
        [Display(Name = "MaxORTicks", GroupName = "Parameters", Order = 14)]
        public int MaxORTicks { get; set; } = 200;

        [NinjaScriptProperty]
        [Display(Name = "UseBiasFilter", GroupName = "Parameters", Order = 15)]
        public bool UseBiasFilter { get; set; } = false;

        [NinjaScriptProperty]
        [Display(Name = "BiasEMAPeriod", GroupName = "Parameters", Order = 16)]
        public int BiasEMAPeriod { get; set; } = 20;

        [NinjaScriptProperty]
        [Display(Name = "UseHTFBias", GroupName = "Parameters", Order = 17)]
        public bool UseHTFBias { get; set; } = false;

        [NinjaScriptProperty]
        [Display(Name = "HTFEMAPeriod", GroupName = "Parameters", Order = 18)]
        public int HTFEMAPeriod { get; set; } = 20;

        [NinjaScriptProperty]
        [Display(Name = "UseDailyLevelsFilter", GroupName = "Parameters", Order = 19)]
        public bool UseDailyLevelsFilter { get; set; } = false;

        [NinjaScriptProperty]
        [Display(Name = "RequireGap", GroupName = "Parameters", Order = 20)]
        public bool RequireGap { get; set; } = false;

        [NinjaScriptProperty]
        [Display(Name = "MinGapTicks", GroupName = "Parameters", Order = 21)]
        public int MinGapTicks { get; set; } = 5;

        [NinjaScriptProperty]
        [Display(Name = "AvoidPriorDayLevels", GroupName = "Parameters", Order = 22)]
        public bool AvoidPriorDayLevels { get; set; } = true;

        [NinjaScriptProperty]
        [Display(Name = "PriorDayLevelBufferATR", GroupName = "Parameters", Order = 23)]
        public double PriorDayLevelBufferATR { get; set; } = 1.0;

        [NinjaScriptProperty]
        [Display(Name = "UseAtrScaling", GroupName = "Parameters", Order = 24)]
        public bool UseAtrScaling { get; set; } = false;

        [NinjaScriptProperty]
        [Display(Name = "RetestBufferATR", GroupName = "Parameters", Order = 25)]
        public double RetestBufferATR { get; set; } = 0.03;

        [NinjaScriptProperty]
        [Display(Name = "MaxDistanceFromORATR", GroupName = "Parameters", Order = 26)]
        public double MaxDistanceFromORATR { get; set; } = 0.10;

        [NinjaScriptProperty]
        [Display(Name = "MinORATR", GroupName = "Parameters", Order = 27)]
        public double MinORATR { get; set; } = 0.03;

        [NinjaScriptProperty]
        [Display(Name = "MaxORATR", GroupName = "Parameters", Order = 28)]
        public double MaxORATR { get; set; } = 0.20;

        [NinjaScriptProperty]
        [Display(Name = "UseVolumeFilter", GroupName = "Parameters", Order = 29)]
        public bool UseVolumeFilter { get; set; } = false;

        [NinjaScriptProperty]
        [Display(Name = "MinBreakoutVolMult", GroupName = "Parameters", Order = 30)]
        public double MinBreakoutVolMult { get; set; } = 1.5;


        // === NEW BIAS PARAMETERS ===

        [NinjaScriptProperty]
        [Display(Name = "UseGapDirectionalBias", GroupName = "Bias", Order = 31)]
        public bool UseGapDirectionalBias { get; set; } = false;

        [NinjaScriptProperty]
        [Display(Name = "UsePDRangeBias", GroupName = "Bias", Order = 32)]
        public bool UsePDRangeBias { get; set; } = false;

        [NinjaScriptProperty]
        [Display(Name = "UseOpening5MinBias", GroupName = "Bias", Order = 33)]
        public bool UseOpening5MinBias { get; set; } = false;

        [NinjaScriptProperty]
        [Display(Name = "OpeningBiasMinutes", GroupName = "Bias", Order = 34)]
        [Range(1, 30)]
        public int OpeningBiasMinutes { get; set; } = 5;



        // === INTERNAL STATE ===

        // OR
        private double openingRangeHigh;
        private double openingRangeLow;
        private bool   rangeComplete;
        private bool   tradeToday;
        private int    tradesToday;

        private DateTime sessionOpenTime = Core.Globals.MinDate;
        private DateTime orEndTime       = Core.Globals.MinDate;
        private DateTime tradeEndTime    = Core.Globals.MinDate;

        // breakout tracking
        private int barsSinceHighBreak;
        private int barsSinceLowBreak;

        // NEW: number of primary bars since session open
        private int barsSinceSessionStart;

        // volume
        private double orVolumeSum;
        private int    orBarCount;
        private double avgOrVolume;

        // bias / HTF
        private EMA ema5Min;
        private EMA emaDaily;
        private EMA emaHourly;
        private ATR atrDaily;

        // multi-series indices
        private int idx5Min   = -1;
        private int idxDaily  = -1;
        private int idxHourly = -1;

        // previous day levels / gap
        private double priorDayHigh;
        private double priorDayLow;
        private double priorDayClose;
        private double sessionOpenPrice;
        private double gapSize;

        // bias flags
        private bool gapUp;
        private bool gapDown;
        private bool pdBiasBull;
        private bool pdBiasBear;

        // opening 5m bias
        private bool openingBiasComputed;
        private bool openingBiasUp;
        private bool openingBiasDown;


        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Name = "OpeningRangeRetestEngulf_Permissive";
                Calculate = Calculate.OnBarClose;
                EntriesPerDirection = 1;
                EntryHandling = EntryHandling.AllEntries;
                IsExitOnSessionCloseStrategy = true;
                ExitOnSessionCloseSeconds = 30;
                IsInstantiatedOnEachOptimizationIteration = false;
            }
            else if (State == State.Configure)
            {
                int nextIdx = 1;

                if (UseBiasFilter)
                {
                    AddDataSeries(BarsPeriodType.Minute, 5);
                    idx5Min = nextIdx++;
                }

                if (UseHTFBias || UseDailyLevelsFilter || UseAtrScaling || UseGapDirectionalBias || UsePDRangeBias)
                {
                    AddDataSeries(BarsPeriodType.Day, 1);
                    idxDaily = nextIdx++;
                }

                if (UseHTFBias || UseDailyLevelsFilter)
                {
                    AddDataSeries(BarsPeriodType.Minute, 60);
                    idxHourly = nextIdx++;
                }
            }
            else if (State == State.DataLoaded)
            {
                if (idx5Min >= 0)
                    ema5Min = EMA(BarsArray[idx5Min], BiasEMAPeriod);

                if (idxDaily >= 0)
                {
                    if (UseAtrScaling || UseDailyLevelsFilter || UseGapDirectionalBias || UsePDRangeBias)
                        atrDaily = ATR(BarsArray[idxDaily], 14);

                    if (UseHTFBias)
                        emaDaily = EMA(BarsArray[idxDaily], HTFEMAPeriod);
                }

                if (idxHourly >= 0 && UseHTFBias)
                    emaHourly = EMA(BarsArray[idxHourly], HTFEMAPeriod);
            }
        }

        private double GetDailyATR()
        {
            if (atrDaily == null || idxDaily < 0)
                return 0;

            if (CurrentBars[idxDaily] < atrDaily.BarsRequiredToPlot)
                return 0;

            return atrDaily[0];
        }

        private void ResetSession()
        {
            openingRangeHigh = double.MinValue;
            openingRangeLow  = double.MaxValue;
            rangeComplete    = false;
            tradeToday       = true;

            tradesToday      = 0;
            barsSinceHighBreak = int.MaxValue;
            barsSinceLowBreak  = int.MaxValue;

            // NEW
            barsSinceSessionStart = 0;

            orVolumeSum = 0;
            orBarCount  = 0;
            avgOrVolume = 0;

            gapUp = gapDown = false;
            pdBiasBull = pdBiasBear = false;

            openingBiasComputed = false;
            openingBiasUp = openingBiasDown = false;

            if (idxDaily >= 0 && CurrentBars[idxDaily] >= 1)
            {
                priorDayHigh  = Highs[idxDaily][1];
                priorDayLow   = Lows[idxDaily][1];
                priorDayClose = Closes[idxDaily][1];
                sessionOpenPrice = Open[0];
                gapSize = sessionOpenPrice - priorDayClose;

                int gapTicks = (int)Math.Round(Math.Abs(gapSize) / TickSize);
                gapUp = gapSize > 0;
                gapDown = gapSize < 0;

                if (Debug)
                {
                    D(string.Format(
                        "Prior Day: H={0:F2} L={1:F2} C={2:F2} | Open={3:F2} Gap={4:F2} ({5} ticks)",
                        priorDayHigh, priorDayLow, priorDayClose, sessionOpenPrice, gapSize, gapTicks));
                }

                if (UsePDRangeBias)
                {
                    double buffer = 2 * TickSize;
                    if (sessionOpenPrice > priorDayHigh + buffer)
                        pdBiasBull = true;
                    else if (sessionOpenPrice < priorDayLow - buffer)
                        pdBiasBear = true;

                    if (Debug)
                    {
                        if (pdBiasBull) D("PD RANGE BIAS: Bullish (open above PDH)");
                        else if (pdBiasBear) D("PD RANGE BIAS: Bearish (open below PDL)");
                        else D("PD RANGE BIAS: Neutral (open inside prior range)");
                    }
                }

                if (RequireGap)
                {
                    if (gapTicks < MinGapTicks)
                    {
                        tradeToday = false;
                        D(string.Format("GAP FILTER: gap too small ({0} ticks < {1}) - no trading", gapTicks, MinGapTicks));
                    }
                }
            }

            if (sessionOpenTime != Core.Globals.MinDate)
            {
                string tagHigh = "ORHigh_" + sessionOpenTime.ToString("yyyyMMdd_HHmm");
                string tagLow  = "ORLow_"  + sessionOpenTime.ToString("yyyyMMdd_HHmm");
                RemoveDrawObject(tagHigh);
                RemoveDrawObject(tagLow);
            }
        }

        protected override void OnBarUpdate()
        {
            if (BarsInProgress != 0)
                return;

            if (CurrentBar < BarsRequiredToTrade)
                return;

            DateTime bt = Times[0][0];

            // === 1) New session handling & bar counter ===
            if (Bars.IsFirstBarOfSession)
            {
                sessionOpenTime = bt;
                orEndTime    = sessionOpenTime.AddMinutes(OpeningRangeMinutes);
                tradeEndTime = (StopTradingMinutesAfterOpen > 0)
                               ? sessionOpenTime.AddMinutes(StopTradingMinutesAfterOpen)
                               : DateTime.MaxValue;

                ResetSession();
                D(string.Format("=== NEW SESSION {0} ===", sessionOpenTime));

                barsSinceSessionStart = 0;   // first bar of session
            }
            else if (sessionOpenTime != Core.Globals.MinDate)
            {
                barsSinceSessionStart++;     // subsequent bars
            }

            if (sessionOpenTime == Core.Globals.MinDate)
                return;

            // === 1b) Opening 5-minute bias based on first N bars ===
            if (UseOpening5MinBias && !openingBiasComputed)
            {
                int biasBars = Math.Max(1, OpeningBiasMinutes);   // 1-min primary
                if (barsSinceSessionStart == biasBars - 1)
                {
                    double openBiasOpen  = Open[biasBars - 1];  // first bar of session
                    double openBiasClose = Close[0];            // this bar

                    double body = openBiasClose - openBiasOpen;
                    double threshold = TickSize * 2;

                    if (body > threshold)
                        openingBiasUp = true;
                    else if (body < -threshold)
                        openingBiasDown = true;

                    openingBiasComputed = true;

                    if (Debug)
                    {
                        string dir = "Neutral";
                        if (openingBiasUp) dir = "Bullish";
                        else if (openingBiasDown) dir = "Bearish";

                        D(string.Format("OPENING {0}m BIAS: {1} (open={2:F2}, close={3:F2})",
                            OpeningBiasMinutes, dir, openBiasOpen, openBiasClose));
                    }
                }
            }

            // === 2) Build opening range ===
            if (!rangeComplete)
            {
                if (bt <= orEndTime)
                {
                    openingRangeHigh = Math.Max(openingRangeHigh, High[0]);
                    openingRangeLow  = Math.Min(openingRangeLow, Low[0]);

                    if (UseVolumeFilter)
                    {
                        orVolumeSum += Volume[0];
                        orBarCount++;
                    }

                    return;
                }
                else if (openingRangeHigh > double.MinValue && openingRangeLow < double.MaxValue)
                {
                    rangeComplete = true;

                    if (UseVolumeFilter && orBarCount > 0)
                        avgOrVolume = orVolumeSum / orBarCount;

                    double orRange = openingRangeHigh - openingRangeLow;
                    int orTicks    = (int)Math.Round(orRange / TickSize);

                    double atrForOR = GetDailyATR();
                    bool orOutOfRange = false;
                    string reason = "";

                    if (UseAtrScaling && atrForOR > 0)
                    {
                        double orToAtr = orRange / atrForOR;
                        if (orToAtr < MinORATR || orToAtr > MaxORATR)
                        {
                            orOutOfRange = true;
                            reason = string.Format("OR/ATR={0:F3} (Min={1:F3}, Max={2:F3})", orToAtr, MinORATR, MaxORATR);
                        }
                    }
                    else
                    {
                        if (orTicks < MinORTicks || orTicks > MaxORTicks)
                        {
                            orOutOfRange = true;
                            reason = string.Format("Ticks={0} (Min={1}, Max={2})", orTicks, MinORTicks, MaxORTicks);
                        }
                    }

                    if (orOutOfRange)
                    {
                        tradeToday = false;
                        D(string.Format("{0} OR OUT OF RANGE, no trading. {1}", bt, reason));
                    }

                    D(string.Format(
                        "{0} OR DONE  High={1:F2}  Low={2:F2}  Range={3:F2}  Ticks={4}  TradeToday={5}  AvgVol={6:F0}",
                        bt, openingRangeHigh, openingRangeLow, orRange, orTicks, tradeToday, avgOrVolume));

                    string tagHigh = "ORHigh_" + sessionOpenTime.ToString("yyyyMMdd_HHmm");
                    string tagLow  = "ORLow_"  + sessionOpenTime.ToString("yyyyMMdd_HHmm");
                    Draw.HorizontalLine(this, tagHigh, openingRangeHigh, Brushes.DodgerBlue);
                    Draw.HorizontalLine(this, tagLow,  openingRangeLow,  Brushes.OrangeRed);
                }
            }

            if (!rangeComplete || !tradeToday)
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

            if (barsSinceHighBreak == int.MaxValue && Close[0] > openingRangeHigh)
            {
                bool validBreakout = true;

                if (UseVolumeFilter && avgOrVolume > 0)
                {
                    double mult = Volume[0] / avgOrVolume;
                    if (mult < MinBreakoutVolMult)
                    {
                        validBreakout = false;
                        D(string.Format("{0} BREAKOUT HIGH REJECTED: Vol={1:F0} AvgVol={2:F0} Mult={3:F2} < {4:F2}",
                            bt, Volume[0], avgOrVolume, mult, MinBreakoutVolMult));
                    }
                    else
                    {
                        D(string.Format("{0} BREAKOUT HIGH  Close={1} ORHigh={2} VolMult={3:F2}",
                            bt, Close[0], openingRangeHigh, mult));
                    }
                }
                else
                {
                    D(string.Format("{0} BREAKOUT HIGH  Close={1} ORHigh={2}", bt, Close[0], openingRangeHigh));
                }

                if (validBreakout)
                    barsSinceHighBreak = 0;
            }

            if (barsSinceLowBreak == int.MaxValue && Close[0] < openingRangeLow)
            {
                bool validBreakout = true;

                if (UseVolumeFilter && avgOrVolume > 0)
                {
                    double mult = Volume[0] / avgOrVolume;
                    if (mult < MinBreakoutVolMult)
                    {
                        validBreakout = false;
                        D(string.Format("{0} BREAKOUT LOW REJECTED: Vol={1:F0} AvgVol={2:F0} Mult={3:F2} < {4:F2}",
                            bt, Volume[0], avgOrVolume, mult, MinBreakoutVolMult));
                    }
                    else
                    {
                        D(string.Format("{0} BREAKOUT LOW   Close={1} ORLow={2} VolMult={3:F2}",
                            bt, Close[0], openingRangeLow, mult));
                    }
                }
                else
                {
                    D(string.Format("{0} BREAKOUT LOW   Close={1} ORLow={2}", bt, Close[0], openingRangeLow));
                }

                if (validBreakout)
                    barsSinceLowBreak = 0;
            }

            if (Position.MarketPosition != MarketPosition.Flat)
                return;

            // === 4) Bias filters ===
            bool biasAllowsLongs  = true;
            bool biasAllowsShorts = true;
            bool htfBullish = true;
            bool htfBearish = false;

            if (UseBiasFilter && ema5Min != null && idx5Min >= 0 &&
                CurrentBars[idx5Min] >= ema5Min.BarsRequiredToPlot)
            {
                double close5 = Closes[idx5Min][0];
                double ema5   = ema5Min[0];

                if (close5 > ema5)
                {
                    biasAllowsLongs  = true;
                    biasAllowsShorts = false;
                }
                else
                {
                    biasAllowsLongs  = false;
                    biasAllowsShorts = true;
                }
            }

            if (UseHTFBias && emaDaily != null && emaHourly != null &&
                idxDaily >= 0 && idxHourly >= 0 &&
                CurrentBars[idxDaily] >= emaDaily.BarsRequiredToPlot &&
                CurrentBars[idxHourly] >= emaHourly.BarsRequiredToPlot)
            {
                double dClose = Closes[idxDaily][0];
                double dEMA   = emaDaily[0];
                double hClose = Closes[idxHourly][0];
                double hEMA   = emaHourly[0];

                bool dBull = dClose > dEMA;
                bool hBull = hClose > hEMA;
                bool dBear = dClose < dEMA;
                bool hBear = hClose < hEMA;

                if (dBull && hBull)
                {
                    htfBullish = true;
                    htfBearish = false;
                }
                else if (dBear && hBear)
                {
                    htfBullish = false;
                    htfBearish = true;
                }
                else
                {
                    htfBullish = false;
                    htfBearish = false;
                }
            }

            bool dailyLevelsAllowLong  = true;
            bool dailyLevelsAllowShort = true;

            if (UseDailyLevelsFilter && priorDayHigh > 0 && priorDayLow > 0 && atrDaily != null && idxDaily >= 0 &&
                CurrentBars[idxDaily] >= atrDaily.BarsRequiredToPlot)
            {
                double atrVal = atrDaily[0];
                double buffer = atrVal * PriorDayLevelBufferATR;

                double longTargetEstimate  = openingRangeHigh + (openingRangeHigh - openingRangeLow) * RewardRiskRatio;
                double shortTargetEstimate = openingRangeLow  - (openingRangeHigh - openingRangeLow) * RewardRiskRatio;

                if (longTargetEstimate >= priorDayHigh - buffer)
                {
                    dailyLevelsAllowLong = false;
                    if (Debug)
                        D(string.Format("{0} DAILY FILTER: Long target {1:F2} near PDH {2:F2}", bt, longTargetEstimate, priorDayHigh));
                }

                if (shortTargetEstimate <= priorDayLow + buffer)
                {
                    dailyLevelsAllowShort = false;
                    if (Debug)
                        D(string.Format("{0} DAILY FILTER: Short target {1:F2} near PDL {2:F2}", bt, shortTargetEstimate, priorDayLow));
                }
            }

            // NEW: gap / PD / opening-bias directional filters
            if (UseGapDirectionalBias)
            {
                if (gapUp)
                    biasAllowsShorts = false;
                else if (gapDown)
                    biasAllowsLongs = false;
            }

            if (UsePDRangeBias)
            {
                if (pdBiasBull)
                    biasAllowsShorts = false;
                else if (pdBiasBear)
                    biasAllowsLongs = false;
            }

            if (UseOpening5MinBias && openingBiasComputed)
            {
                if (openingBiasUp)
                    biasAllowsShorts = false;
                else if (openingBiasDown)
                    biasAllowsLongs = false;
            }

            double atr = GetDailyATR();

            double bufferRetest = RetestBufferTicks * TickSize;
            if (UseAtrScaling && atr > 0)
                bufferRetest = RetestBufferATR * atr;

            // === 5) LONG SETUP ===
            bool longSignal = false;
            if (barsSinceHighBreak >= 0 && barsSinceHighBreak <= MaxBarsAfterBreakout)
            {
                double distance = Math.Abs(Close[0] - openingRangeHigh);
                double maxDist  = MaxDistanceFromOR * TickSize;
                if (UseAtrScaling && atr > 0)
                    maxDist = MaxDistanceFromORATR * atr;

                bool nearLevel    = distance <= maxDist;
                bool touchedLevel = Low[0] <= openingRangeHigh + bufferRetest &&
                                    High[0] >= openingRangeHigh - bufferRetest;

                bool bullishEngulf = IsBullishEngulfing(0);
                bool bullishReject = IsBullishRejection(0);

                if (touchedLevel)
                    D(string.Format("{0} LONG touch  Close={1} Low={2} ORHigh={3}", bt, Close[0], Low[0], openingRangeHigh));
                if (bullishEngulf)
                    D(string.Format("{0} LONG engulf candidate", bt));
                if (bullishReject)
                    D(string.Format("{0} LONG rejection candidate", bt));

                bool levelCondition = RequireRetest ? touchedLevel : nearLevel;

                if (levelCondition && (bullishEngulf || (!UseEngulfOnly && bullishReject)))
                    longSignal = true;
            }

            // === 6) SHORT SETUP ===
            bool shortSignal = false;
            if (barsSinceLowBreak >= 0 && barsSinceLowBreak <= MaxBarsAfterBreakout)
            {
                double distance = Math.Abs(Close[0] - openingRangeLow);
                double maxDist  = MaxDistanceFromOR * TickSize;
                if (UseAtrScaling && atr > 0)
                    maxDist = MaxDistanceFromORATR * atr;

                bool nearLevel    = distance <= maxDist;
                bool touchedLevel = High[0] >= openingRangeLow - bufferRetest &&
                                    Low[0]  <= openingRangeLow + bufferRetest;

                bool bearishEngulf = IsBearishEngulfing(0);
                bool bearishReject = IsBearishRejection(0);

                if (touchedLevel)
                    D(string.Format("{0} SHORT touch  Close={1} High={2} ORLow={3}", bt, Close[0], High[0], openingRangeLow));
                if (bearishEngulf)
                    D(string.Format("{0} SHORT engulf candidate", bt));
                if (bearishReject)
                    D(string.Format("{0} SHORT rejection candidate", bt));

                bool levelCondition = RequireRetest ? touchedLevel : nearLevel;

                if (levelCondition && (bearishEngulf || (!UseEngulfOnly && bearishReject)))
                    shortSignal = true;
            }

            // === 7) EXECUTION GATING ===
            bool canTradeLong  = EnableLongs  && biasAllowsLongs  && dailyLevelsAllowLong;
            bool canTradeShort = EnableShorts && biasAllowsShorts && dailyLevelsAllowShort;

            if (UseHTFBias)
            {
                canTradeLong  &= htfBullish;
                canTradeShort &= htfBearish;
            }

            if (canTradeLong && longSignal)
            {
                double entry = Close[0];
                double stop  = Low[0];
                double risk  = entry - stop;

                if (risk <= TickSize * 2)
                    return;

                double target = entry + risk * RewardRiskRatio;

                SetStopLoss(CalculationMode.Price, stop);
                SetProfitTarget(CalculationMode.Price, target);

                D(string.Format("{0} ENTER LONG  entry={1} stop={2} target={3}", bt, entry, stop, target));
                EnterLong();
                tradesToday++;
                return;
            }

            if (canTradeShort && shortSignal)
            {
                double entry = Close[0];
                double stop  = High[0];
                double risk  = stop - entry;

                if (risk <= TickSize * 2)
                    return;

                double target = entry - risk * RewardRiskRatio;

                SetStopLoss(CalculationMode.Price, stop);
                SetProfitTarget(CalculationMode.Price, target);

                D(string.Format("{0} ENTER SHORT entry={1} stop={2} target={3}", bt, entry, stop, target));
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

            if (Close[idx + 1] >= Open[idx + 1])
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

            if (Close[idx + 1] <= Open[idx + 1])
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

        private void D(string msg)
        {
            if (Debug)
                Print(msg);
        }
    }
}
