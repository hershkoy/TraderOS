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
        public int OpeningRangeMinutes { get; set; } = 15;  // default to 15m per transcripts

        [NinjaScriptProperty]
        [Display(Name = "StopTradingMinutesAfterOpen", GroupName = "Parameters", Order = 6)]
        public int StopTradingMinutesAfterOpen { get; set; } = 180;   // e.g. first 3 hours

        [NinjaScriptProperty]
        [Display(Name = "Debug", GroupName = "Parameters", Order = 7)]
        public bool Debug { get; set; } = true;

        [NinjaScriptProperty]
        [Display(Name = "RetestBufferTicks", GroupName = "Parameters", Order = 8)]
        public int RetestBufferTicks { get; set; } = 3;

        [NinjaScriptProperty]
        [Display(Name = "RequireRetest", GroupName = "Parameters", Order = 9)]
        public bool RequireRetest { get; set; } = false;   // more permissive by default

        [NinjaScriptProperty]
        [Display(Name = "MaxDistanceFromOR", GroupName = "Parameters", Order = 10)]
        public int MaxDistanceFromOR { get; set; } = 20;   // in ticks

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
        public double RetestBufferATR { get; set; } = 0.03;  // 3% of ATR

        [NinjaScriptProperty]
        [Display(Name = "MaxDistanceFromORATR", GroupName = "Parameters", Order = 26)]
        public double MaxDistanceFromORATR { get; set; } = 0.10;  // 10% of ATR

        [NinjaScriptProperty]
        [Display(Name = "MinORATR", GroupName = "Parameters", Order = 27)]
        public double MinORATR { get; set; } = 0.03;  // 3% of ATR

        [NinjaScriptProperty]
        [Display(Name = "MaxORATR", GroupName = "Parameters", Order = 28)]
        public double MaxORATR { get; set; } = 0.20;  // 20% of ATR

        [NinjaScriptProperty]
        [Display(Name = "UseVolumeFilter", GroupName = "Parameters", Order = 29)]
        public bool UseVolumeFilter { get; set; } = false;

        [NinjaScriptProperty]
        [Display(Name = "MinBreakoutVolMult", GroupName = "Parameters", Order = 30)]
        public double MinBreakoutVolMult { get; set; } = 1.5;  // 1.5x average OR volume


        // === INTERNAL STATE ===

        private double openingRangeHigh;
        private double openingRangeLow;
        private bool   rangeComplete;
        private bool   tradeToday;   // flag to skip trading if OR is out of range

        private int tradesToday;

        private DateTime sessionOpenTime = Core.Globals.MinDate;
        private DateTime orEndTime       = Core.Globals.MinDate;
        private DateTime tradeEndTime    = Core.Globals.MinDate;

        // breakout trackers: int.MaxValue means "no breakout yet"
        private int barsSinceHighBreak;
        private int barsSinceLowBreak;

        // multi-timeframe bias filter (legacy 5-min)
        private EMA ema5Min;

        // higher timeframe bias (daily and hourly)
        private EMA emaDaily;
        private EMA emaHourly;
        private ATR atrDaily;

        // previous day levels
        private double priorDayHigh;
        private double priorDayLow;
        private double priorDayClose;
        private double sessionOpenPrice;
        private double gapSize;

        // OR volume tracking
        private double orVolumeSum;
        private int orBarCount;
        private double avgOrVolume;


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

                RewardRiskRatio = 2.0;
                MaxTradesPerDay = 2;
                MaxBarsAfterBreakout = 20;
                UseEngulfOnly = false;
                RejectionFactor = 1.5;
                OpeningRangeMinutes = 15;
                StopTradingMinutesAfterOpen = 180;
                Debug = true;

                RetestBufferTicks = 3;
                RequireRetest = false;
                MaxDistanceFromOR = 20;
                EnableLongs = true;
                EnableShorts = true;
                MinORTicks = 10;
                MaxORTicks = 200;
                UseBiasFilter = false;
                BiasEMAPeriod = 20;
                UseHTFBias = false;
                HTFEMAPeriod = 20;
                UseDailyLevelsFilter = false;
                RequireGap = false;
                MinGapTicks = 5;
                AvoidPriorDayLevels = true;
                PriorDayLevelBufferATR = 1.0;
                UseAtrScaling = false;
                RetestBufferATR = 0.03;
                MaxDistanceFromORATR = 0.10;
                MinORATR = 0.03;
                MaxORATR = 0.20;
                UseVolumeFilter = false;
                MinBreakoutVolMult = 1.5;
            }
            else if (State == State.Configure)
            {
                // Add 5-minute data series for legacy bias filter
                if (UseBiasFilter)
                {
                    AddDataSeries(BarsPeriodType.Minute, 5);
                }

                // Add daily and hourly data series for HTF bias, daily levels, or ATR scaling
                if (UseHTFBias || UseDailyLevelsFilter || UseAtrScaling)
                {
                    AddDataSeries(BarsPeriodType.Day, 1);
                    if (UseHTFBias || UseDailyLevelsFilter)
                    {
                        AddDataSeries(BarsPeriodType.Minute, 60);  // 1-hour (only if needed for HTF bias)
                    }
                }
            }
            else if (State == State.DataLoaded)
            {
                // Calculate data series indices based on what's enabled
                // Order: primary[0], 5min[1 if UseBiasFilter], daily[1 or 2], hourly[2 or 3]
                int idx5Min = -1;
                int idxDaily = -1;
                int idxHourly = -1;

                int currentIdx = 1;
                if (UseBiasFilter)
                {
                    idx5Min = currentIdx++;
                }
                if (UseHTFBias || UseDailyLevelsFilter || UseAtrScaling)
                {
                    idxDaily = currentIdx++;
                    if (UseHTFBias || UseDailyLevelsFilter)
                    {
                        idxHourly = currentIdx++;
                    }
                }

                // Initialize EMA on 5-minute series (legacy)
                if (UseBiasFilter && idx5Min >= 0 && BarsArray.Length > idx5Min)
                {
                    ema5Min = EMA(BarsArray[idx5Min], BiasEMAPeriod);
                }

                // Initialize HTF indicators and ATR
                if (idxDaily >= 0 && BarsArray.Length > idxDaily)
                {
                    // Always initialize ATR if daily series is available (needed for ATR scaling or daily levels)
                    if (UseAtrScaling || UseDailyLevelsFilter)
                    {
                        atrDaily = ATR(BarsArray[idxDaily], 14);
                    }

                    // Initialize daily EMA if HTF bias is enabled
                    if (UseHTFBias)
                    {
                        emaDaily = EMA(BarsArray[idxDaily], HTFEMAPeriod);
                    }
                }

                // Initialize hourly EMA if HTF bias is enabled
                if ((UseHTFBias || UseDailyLevelsFilter) && idxHourly >= 0 && BarsArray.Length > idxHourly)
                {
                    emaHourly = EMA(BarsArray[idxHourly], HTFEMAPeriod);
                }
            }
        }

        private void ResetSession()
        {
            openingRangeHigh = double.MinValue;
            openingRangeLow  = double.MaxValue;
            rangeComplete    = false;
            tradeToday       = true;   // default to trading, will be set when OR completes

            tradesToday      = 0;

            barsSinceHighBreak = int.MaxValue;
            barsSinceLowBreak  = int.MaxValue;

            // Reset OR volume tracking
            orVolumeSum = 0;
            orBarCount = 0;
            avgOrVolume = 0;

            // Load previous day levels
            if (UseDailyLevelsFilter && BarsArray.Length > 1)
            {
                // Daily index: comes after 5-min if UseBiasFilter is enabled
                int idxDaily = (UseBiasFilter && (UseHTFBias || UseDailyLevelsFilter || UseAtrScaling)) ? 2 : 
                               ((UseHTFBias || UseDailyLevelsFilter || UseAtrScaling) ? 1 : -1);
                if (idxDaily >= 0 && BarsArray.Length > idxDaily && CurrentBars[idxDaily] >= 1)
                {
                    priorDayHigh = Highs[idxDaily][1];
                    priorDayLow = Lows[idxDaily][1];
                    priorDayClose = Closes[idxDaily][1];
                    sessionOpenPrice = Open[0];
                    gapSize = sessionOpenPrice - priorDayClose;

                    D(string.Format("Prior Day: High={0:F2} Low={1:F2} Close={2:F2} | Gap={3:F2} ({4} ticks)",
                        priorDayHigh, priorDayLow, priorDayClose, gapSize, (int)Math.Round(Math.Abs(gapSize) / TickSize)));
                }
            }

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
                    
                    // Track volume during OR building
                    if (UseVolumeFilter)
                    {
                        orVolumeSum += Volume[0];
                        orBarCount++;
                    }
                    
                    return; // still building range
                }
                else if (openingRangeHigh > double.MinValue && openingRangeLow < double.MaxValue)
                {
                    rangeComplete = true;

                    // Calculate average OR volume
                    if (UseVolumeFilter && orBarCount > 0)
                    {
                        avgOrVolume = orVolumeSum / orBarCount;
                    }

                    double orRange = openingRangeHigh - openingRangeLow;
                    int orTicks = (int)Math.Round(orRange / TickSize);

                    // Get ATR for scaling if enabled
                    double atrForOR = 0;
                    if ((UseAtrScaling || UseDailyLevelsFilter) && atrDaily != null)
                    {
                        int idxDaily = (UseBiasFilter && (UseHTFBias || UseDailyLevelsFilter || UseAtrScaling)) ? 2 : 
                                       ((UseHTFBias || UseDailyLevelsFilter || UseAtrScaling) ? 1 : -1);
                        if (idxDaily >= 0 && BarsArray.Length > idxDaily && CurrentBars[idxDaily] >= atrDaily.BarsRequiredToPlot)
                        {
                            atrForOR = atrDaily[0];
                        }
                    }

                    // Check if OR is within acceptable range (tick-based or ATR-based)
                    tradeToday = true;
                    bool orOutOfRange = false;
                    string orRejectReason = "";

                    if (UseAtrScaling && atrForOR > 0)
                    {
                        // ATR-based OR filter
                        double orToAtr = orRange / atrForOR;
                        if (orToAtr < MinORATR || orToAtr > MaxORATR)
                        {
                            orOutOfRange = true;
                            orRejectReason = string.Format("OR/ATR={0:F3} (Min={1:F3}, Max={2:F3})", orToAtr, MinORATR, MaxORATR);
                        }
                    }
                    else
                    {
                        // Tick-based OR filter (legacy)
                        if (orTicks < MinORTicks || orTicks > MaxORTicks)
                        {
                            orOutOfRange = true;
                            orRejectReason = string.Format("Ticks={0} (Min={1}, Max={2})", orTicks, MinORTicks, MaxORTicks);
                        }
                    }

                    if (orOutOfRange)
                    {
                        tradeToday = false;
                        D(string.Format("{0} OR OUT OF RANGE, no trading. {1}", bt, orRejectReason));
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

            if (!rangeComplete)
                return;

            if (!tradeToday)
                return;

            if (bt > tradeEndTime)
                return;

            if (tradesToday >= MaxTradesPerDay)
                return;

            // === 3) Track breakout state (first breakout only) ===
            if (barsSinceHighBreak != int.MaxValue)
                barsSinceHighBreak++;
            if (barsSinceLowBreak != int.MaxValue)
                barsSinceLowBreak++;

            // First close above OR high -> mark breakout (with volume filter if enabled)
            if (barsSinceHighBreak == int.MaxValue && Close[0] > openingRangeHigh)
            {
                bool validBreakout = true;
                
                if (UseVolumeFilter && avgOrVolume > 0)
                {
                    double volMult = Volume[0] / avgOrVolume;
                    if (volMult < MinBreakoutVolMult)
                    {
                        validBreakout = false;
                        if (Debug)
                            D(string.Format("{0} BREAKOUT HIGH REJECTED: Volume={1:F0} AvgVol={2:F0} Mult={3:F2} < {4:F2}",
                                bt, Volume[0], avgOrVolume, volMult, MinBreakoutVolMult));
                    }
                    else
                    {
                        if (Debug)
                            D(string.Format("{0} BREAKOUT HIGH  Close={1} ORHigh={2} VolMult={3:F2}", 
                                bt, Close[0], openingRangeHigh, volMult));
                    }
                }
                else
                {
                    D(string.Format("{0} BREAKOUT HIGH  Close={1} ORHigh={2}", bt, Close[0], openingRangeHigh));
                }

                if (validBreakout)
                    barsSinceHighBreak = 0;
            }

            // First close below OR low -> mark breakout (with volume filter if enabled)
            if (barsSinceLowBreak == int.MaxValue && Close[0] < openingRangeLow)
            {
                bool validBreakout = true;
                
                if (UseVolumeFilter && avgOrVolume > 0)
                {
                    double volMult = Volume[0] / avgOrVolume;
                    if (volMult < MinBreakoutVolMult)
                    {
                        validBreakout = false;
                        if (Debug)
                            D(string.Format("{0} BREAKOUT LOW REJECTED: Volume={1:F0} AvgVol={2:F0} Mult={3:F2} < {4:F2}",
                                bt, Volume[0], avgOrVolume, volMult, MinBreakoutVolMult));
                    }
                    else
                    {
                        if (Debug)
                            D(string.Format("{0} BREAKOUT LOW   Close={1} ORLow={2} VolMult={3:F2}", 
                                bt, Close[0], openingRangeLow, volMult));
                    }
                }
                else
                {
                    D(string.Format("{0} BREAKOUT LOW   Close={1} ORLow={2}", bt, Close[0], openingRangeLow));
                }

                if (validBreakout)
                    barsSinceLowBreak = 0;
            }

            // Only when flat
            if (Position.MarketPosition != MarketPosition.Flat)
                return;

            // === BIAS FILTER: Check higher timeframe EMA alignment ===
            bool biasAllowsLongs = true;
            bool biasAllowsShorts = true;
            bool htfBullish = true;
            bool htfBearish = false;

            // Legacy 5-minute bias filter
            if (UseBiasFilter && ema5Min != null)
            {
                // Calculate 5-min index: it's always index 1 if UseBiasFilter is enabled
                int idx5Min = 1;
                if (BarsArray.Length > idx5Min && CurrentBars[idx5Min] >= ema5Min.BarsRequiredToPlot)
                {
                    double close5Min = Closes[idx5Min][0];
                    double emaValue = ema5Min[0];

                    if (close5Min > emaValue)
                    {
                        biasAllowsLongs = true;
                        biasAllowsShorts = false;
                    }
                    else
                    {
                        biasAllowsLongs = false;
                        biasAllowsShorts = true;
                    }
                }
            }

            // Higher timeframe bias (daily + hourly)
            if (UseHTFBias && emaDaily != null && emaHourly != null)
            {
                // Calculate indices: daily and hourly come after 5-min if it exists
                int idxDaily = (UseBiasFilter && (UseHTFBias || UseDailyLevelsFilter || UseAtrScaling)) ? 2 : 
                               ((UseHTFBias || UseDailyLevelsFilter || UseAtrScaling) ? 1 : -1);
                int idxHourly = (UseBiasFilter && (UseHTFBias || UseDailyLevelsFilter)) ? 3 : 
                                ((UseHTFBias || UseDailyLevelsFilter) ? 2 : -1);

                if (idxDaily >= 0 && idxHourly >= 0 &&
                    CurrentBars[idxDaily] >= emaDaily.BarsRequiredToPlot &&
                    CurrentBars[idxHourly] >= emaHourly.BarsRequiredToPlot)
                {
                    double dailyClose = Closes[idxDaily][0];
                    double dailyEMA = emaDaily[0];
                    double hourlyClose = Closes[idxHourly][0];
                    double hourlyEMA = emaHourly[0];

                    // Both daily and hourly must agree for bias
                    bool dailyBullish = dailyClose > dailyEMA;
                    bool hourlyBullish = hourlyClose > hourlyEMA;
                    bool dailyBearish = dailyClose < dailyEMA;
                    bool hourlyBearish = hourlyClose < hourlyEMA;

                    if (dailyBullish && hourlyBullish)
                    {
                        htfBullish = true;
                        htfBearish = false;
                        if (Debug && (CurrentBar % 10 == 0))
                            D(string.Format("{0} HTF BIAS: Bullish (Daily: {1:F2}>{2:F2}, Hourly: {3:F2}>{4:F2})",
                                bt, dailyClose, dailyEMA, hourlyClose, hourlyEMA));
                    }
                    else if (dailyBearish && hourlyBearish)
                    {
                        htfBullish = false;
                        htfBearish = true;
                        if (Debug && (CurrentBar % 10 == 0))
                            D(string.Format("{0} HTF BIAS: Bearish (Daily: {1:F2}<{2:F2}, Hourly: {3:F2}<{4:F2})",
                                bt, dailyClose, dailyEMA, hourlyClose, hourlyEMA));
                    }
                    else
                    {
                        // Mixed signals - no bias, no trades
                        htfBullish = false;
                        htfBearish = false;
                        if (Debug && (CurrentBar % 10 == 0))
                            D(string.Format("{0} HTF BIAS: Mixed - no trades", bt));
                    }
                }
            }

            // === DAILY LEVELS FILTER ===
            bool dailyLevelsAllowLong = true;
            bool dailyLevelsAllowShort = true;

            if (UseDailyLevelsFilter && priorDayHigh > 0 && priorDayLow > 0)
            {
                // Gap requirement
                if (RequireGap)
                {
                    int gapTicks = (int)Math.Round(Math.Abs(gapSize) / TickSize);
                    if (gapTicks < MinGapTicks)
                    {
                        dailyLevelsAllowLong = false;
                        dailyLevelsAllowShort = false;
                        if (Debug)
                            D(string.Format("{0} DAILY FILTER: Gap too small ({1} ticks < {2})", bt, gapTicks, MinGapTicks));
                    }
                }

                // Avoid trading into prior day's high/low
                int idxDaily = (UseBiasFilter && (UseHTFBias || UseDailyLevelsFilter || UseAtrScaling)) ? 2 : 
                               ((UseHTFBias || UseDailyLevelsFilter || UseAtrScaling) ? 1 : -1);
                if (AvoidPriorDayLevels && atrDaily != null && idxDaily >= 0 && BarsArray.Length > idxDaily && CurrentBars[idxDaily] >= atrDaily.BarsRequiredToPlot)
                {
                    double atrValue = atrDaily[0];
                    double priorDayBuffer = atrValue * PriorDayLevelBufferATR;

                    // For long: check if target would hit prior day high
                    double longTargetEstimate = openingRangeHigh + (openingRangeHigh - openingRangeLow) * RewardRiskRatio;
                    if (longTargetEstimate >= priorDayHigh - priorDayBuffer)
                    {
                        dailyLevelsAllowLong = false;
                        if (Debug)
                            D(string.Format("{0} DAILY FILTER: Long target ({1:F2}) too close to prior high ({2:F2})",
                                bt, longTargetEstimate, priorDayHigh));
                    }

                    // For short: check if target would hit prior day low
                    double shortTargetEstimate = openingRangeLow - (openingRangeHigh - openingRangeLow) * RewardRiskRatio;
                    if (shortTargetEstimate <= priorDayLow + priorDayBuffer)
                    {
                        dailyLevelsAllowShort = false;
                        if (Debug)
                            D(string.Format("{0} DAILY FILTER: Short target ({1:F2}) too close to prior low ({2:F2})",
                                bt, shortTargetEstimate, priorDayLow));
                    }
                }
            }

            // Get ATR for scaling calculations
            double atr = 0;
            if ((UseAtrScaling || UseDailyLevelsFilter) && atrDaily != null)
            {
                int idxDaily = (UseBiasFilter && (UseHTFBias || UseDailyLevelsFilter || UseAtrScaling)) ? 2 : 
                               ((UseHTFBias || UseDailyLevelsFilter || UseAtrScaling) ? 1 : -1);
                if (idxDaily >= 0 && BarsArray.Length > idxDaily && CurrentBars[idxDaily] >= atrDaily.BarsRequiredToPlot)
                {
                    atr = atrDaily[0];
                }
            }

            // buffer for retest zone (ATR-scaled or tick-based)
            double buffer = RetestBufferTicks * TickSize;
            if (UseAtrScaling && atr > 0)
            {
                buffer = RetestBufferATR * atr;
            }

            // === 4) LONG SETUP (more permissive) ===
            bool longSignal = false;
            if (barsSinceHighBreak >= 0 && barsSinceHighBreak <= MaxBarsAfterBreakout)
            {
                // distance from OR high (ATR-scaled or tick-based)
                double distance = Math.Abs(Close[0] - openingRangeHigh);
                double maxDist = MaxDistanceFromOR * TickSize;
                if (UseAtrScaling && atr > 0)
                {
                    maxDist = MaxDistanceFromORATR * atr;
                }
                bool nearLevel = distance <= maxDist;

                // strict retest: bar trades through OR-high zone (with buffer)
                bool touchedLevel    = Low[0] <= openingRangeHigh + buffer &&
                                       High[0] >= openingRangeHigh - buffer;

                bool bullishEngulf   = IsBullishEngulfing(0);
                bool bullishReject   = IsBullishRejection(0);

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

            // === 5) SHORT SETUP (more permissive) ===
            bool shortSignal = false;
            if (barsSinceLowBreak >= 0 && barsSinceLowBreak <= MaxBarsAfterBreakout)
            {
                // distance from OR low (ATR-scaled or tick-based)
                double distance = Math.Abs(Close[0] - openingRangeLow);
                double maxDist = MaxDistanceFromOR * TickSize;
                if (UseAtrScaling && atr > 0)
                {
                    maxDist = MaxDistanceFromORATR * atr;
                }
                bool nearLevel = distance <= maxDist;

                bool touchedLevel    = High[0] >= openingRangeLow - buffer &&
                                       Low[0]  <= openingRangeLow + buffer;

                bool bearishEngulf   = IsBearishEngulfing(0);
                bool bearishReject   = IsBearishRejection(0);

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

            // === 6) EXECUTE ENTRIES ===
            // Combine all filters: manual enable, bias (legacy or HTF), daily levels
            bool canTradeLong = EnableLongs && biasAllowsLongs && dailyLevelsAllowLong;
            if (UseHTFBias)
                canTradeLong = canTradeLong && htfBullish;

            bool canTradeShort = EnableShorts && biasAllowsShorts && dailyLevelsAllowShort;
            if (UseHTFBias)
                canTradeShort = canTradeShort && htfBearish;

            if (canTradeLong && longSignal)
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

            if (canTradeShort && shortSignal)
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

            // previous bar must be bearish
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

            // previous bar must be bullish
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

        // === DEBUG PRINT ===
        private void D(string msg)
        {
            if (Debug)
                Print(msg);
        }
    }
}
