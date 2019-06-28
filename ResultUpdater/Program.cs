using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using Excel = Microsoft.Office.Interop.Excel;

namespace ResultUpdater
{
    class Program
    {
        static Dictionary<string, Tuple<int, string>> dir2feature = new Dictionary<string, Tuple<int, string>>();

        static int expsnum = 10;
        static int classnum = 5;


        static Dictionary<string, int> method2expsnum = new Dictionary<string, int>();

        static void Main(string[] args)
        {
            method2expsnum.Add("svm", 1);
            method2expsnum.Add("rf", 5);
            method2expsnum.Add("dnn", 5);
            method2expsnum.Add("cnn", 5);
            method2expsnum.Add("gru", 5);
            method2expsnum.Add("lstm", 5);

            dir2feature.Add("raw", new Tuple<int, string>(0, "raw"));
            dir2feature.Add("rrintervals", new Tuple<int, string>(1, "rrintervals"));
            dir2feature.Add("fft35", new Tuple<int, string>(2, "fft"));
            dir2feature.Add("waveletdb1lvl3", new Tuple<int, string>(3, "wavelet"));
            dir2feature.Add("waveletdb1lvl3uniformlbp", new Tuple<int, string>(4, "waveletulbp"));
            dir2feature.Add("hos", new Tuple<int, string>(5, "hos"));
            dir2feature.Add("hermite", new Tuple<int, string>(6, "hermite"));


            string imbalance_dir = @"F:\GC_ECG\ECG_SG\ECG\models\imbalanced\";
            string imbalance_path = @"F:\GC_ECG\ECG_SG\ECG\Results_imbalanced.xls";
            GenerateResults(imbalance_dir, imbalance_path);
            string balance_dir = @"F:\GC_ECG\ECG_SG\ECG\models\balanced\";
            string balance_path = @"F:\GC_ECG\ECG_SG\ECG\Results_balanced.xls";
            GenerateResults(balance_dir, balance_path);
        }

        static void GenerateResults(string resultRoot, string resultpath)
        {
            int window = 180;

            var methoddirs = Directory.EnumerateDirectories(resultRoot).ToList();

            Dictionary<string, List<Result>> rets = new Dictionary<string, List<Result>>();
            foreach (var methodDir in methoddirs)
            {
                var methodname = Path.GetFileNameWithoutExtension(methodDir).ToLowerInvariant();
                var subrets = QueryResults(methodDir, method2expsnum[methodname], methodname);
                rets.Add(methodname, subrets);
            }

            int retcount = rets.Sum(f => f.Value.Count());

            List<string> methods = rets.Keys.ToList();

            var xlApp = new Excel.Application();
            if (xlApp == null)
            {
                Console.WriteLine("Excel is not properly installed!!");
                return;
            }

            xlApp.DisplayAlerts = false;
            object misValue = System.Reflection.Missing.Value;

            Excel.Workbook xlWorkBook;
            xlWorkBook = xlApp.Workbooks.Add(misValue);

            Excel.Worksheet xlMajorWorkSheet;
            xlMajorWorkSheet = (Excel.Worksheet)xlWorkBook.Worksheets.Add(xlWorkBook.Worksheets[1]);

            xlMajorWorkSheet.Name = "Summary";
            xlMajorWorkSheet.Cells[1, 1] = "AvgCSR";

            xlMajorWorkSheet.Cells[2, 1] = "Method";

            for (int mi =0; mi< methods.Count; mi++)
            {
                xlMajorWorkSheet.Cells[3 + mi, 1] = methods[mi];
            }

            int sheetcounter = 0;

            int avgtestspeedrow = 3 + methods.Count + 3;
            int avgparamnumrow = avgtestspeedrow + methods.Count + 3;

            xlMajorWorkSheet.Cells[avgtestspeedrow - 2, 1] = "AvgTestSpeedPerSample";
            xlMajorWorkSheet.Cells[avgparamnumrow - 2, 1] = "AvgModelParamNum";

            xlMajorWorkSheet.Cells[avgtestspeedrow - 2, 2] = "Miliseconds";
            xlMajorWorkSheet.Cells[avgparamnumrow - 2, 2] = "Equivalent Number of Float32";

            xlMajorWorkSheet.Cells[avgtestspeedrow - 1, 1] = "Method";
            xlMajorWorkSheet.Cells[avgparamnumrow - 1, 1] = "Method";

            for (int mi = 0; mi < methods.Count; mi++)
            {
                xlMajorWorkSheet.Cells[avgtestspeedrow + mi, 1] = methods[mi];
                xlMajorWorkSheet.Cells[avgparamnumrow + mi, 1] = methods[mi];
            }

            xlMajorWorkSheet.Cells[2, 2] = "raw";
            xlMajorWorkSheet.Cells[2, 3] = "rrintervals";
            xlMajorWorkSheet.Cells[2, 4] = "fft";
            xlMajorWorkSheet.Cells[2, 5] = "wavelet";
            xlMajorWorkSheet.Cells[2, 6] = "waveletulbp";
            xlMajorWorkSheet.Cells[2, 7] = "hos";
            xlMajorWorkSheet.Cells[2, 8] = "hermite";

            for (int mi = 0; mi < methods.Count; mi++)
            {
                var curmethod = methods[mi];
                var currets = rets[curmethod].Select(r =>new Tuple<int, double>(r.Feature.Item1, r.AverageCSR));

                foreach (var cret in currets)
                {
                    xlMajorWorkSheet.Cells[3 + mi, 2 + cret.Item1] = cret.Item2;
                }
            }

            for (int mi = 0; mi < methods.Count; mi++)
            {
                var curmethod = methods[mi];
                var currets = rets[curmethod].Select(r => new Tuple<int, double>(r.Feature.Item1, r.AverageTestSpeed));

                foreach (var cret in currets)
                {
                    xlMajorWorkSheet.Cells[avgtestspeedrow + mi, 2 + cret.Item1] = cret.Item2;
                }
            }

            for (int mi = 0; mi < methods.Count; mi++)
            {
                var curmethod = methods[mi];
                var currets = rets[curmethod].Select(r => new Tuple<int, double>(r.Feature.Item1, r.AverageParamCount));

                foreach (var cret in currets)
                {
                    xlMajorWorkSheet.Cells[avgparamnumrow + mi, 2 + cret.Item1] = cret.Item2;
                }
            }


            Excel.Range numrange = xlMajorWorkSheet.Range[xlMajorWorkSheet.Cells[3, 2], xlMajorWorkSheet.Cells[3 + methods.Count - 1, 2 + dir2feature.Count]];
            numrange.NumberFormat = "0.0%";

            foreach (var iret in rets)
            {
                foreach (var sret in iret.Value)
                {
                    WriteSheet(sret, window, xlWorkBook);

                    sheetcounter++;

                    Console.Title = string.Format("{0}/{1}", sheetcounter, retcount);
                }
            }


            Marshal.FinalReleaseComObject(numrange);

            xlMajorWorkSheet.Columns.AutoFit();
            xlMajorWorkSheet.UsedRange.HorizontalAlignment = Excel.XlHAlign.xlHAlignCenter;


            xlMajorWorkSheet.Move(((Excel.Worksheet)xlWorkBook.Worksheets[1]));


            Marshal.ReleaseComObject(xlMajorWorkSheet);

            ((Excel.Worksheet)xlWorkBook.Worksheets[xlWorkBook.Worksheets.Count]).Delete();


            xlWorkBook.SaveAs(resultpath, Excel.XlFileFormat.xlWorkbookNormal, misValue, misValue, misValue, misValue, Excel.XlSaveAsAccessMode.xlExclusive, misValue, misValue, misValue, misValue, misValue);
            xlWorkBook.Close(true, misValue, misValue);
            xlApp.Quit();

            Marshal.ReleaseComObject(xlWorkBook);
            Marshal.ReleaseComObject(xlApp);
        }

        public static void WriteSheet(Result ret, int window, Excel.Workbook xlWorkBook)
        {
            Excel.Worksheet xlWorkSheet;
            object misValue = System.Reflection.Missing.Value;
            xlWorkSheet = (Excel.Worksheet)xlWorkBook.Worksheets.Add();
            xlWorkSheet.Name = "win_" + window.ToString() + "_" + ret.Name;

            xlWorkSheet.Cells[1, 1] = "Method";
            xlWorkSheet.Cells[1, 2] = ret.Name;

            xlWorkSheet.Cells[2, 1] = "Window";
            xlWorkSheet.Cells[2, 2] = window;

            xlWorkSheet.Cells[3, 1] = "AvgCSR";
            xlWorkSheet.Cells[3, 2] = ret.AverageCSR;

            xlWorkSheet.Cells[5, 1] = "Class";
            xlWorkSheet.Cells[5, 2] = "Recall";
            xlWorkSheet.Cells[5, 3] = "Precision";
            xlWorkSheet.Cells[5, 4] = "F1-Score";

            xlWorkSheet.Cells[5, 6] = "Confusion";

            for (int i = 0; i < classnum; i++)
            {
                xlWorkSheet.Cells[6 + i, 1] = i.ToString();
            }

            for (int i = 1; i <= classnum; i++)
            {
                for (int j = 1; j <= classnum; j++)
                {
                    xlWorkSheet.Cells[5 + i, 5 + j] = ret.Confusion[i][j];
                }
            }

            for (int i = 1; i <= classnum; i++)
            {
                xlWorkSheet.Cells[5 + i, 2] = ret.Metric[i].Recall;
                xlWorkSheet.Cells[5 + i, 3] = ret.Metric[i].Precision;
                xlWorkSheet.Cells[5 + i, 4] = ret.Metric[i].F1Score;
            }

            xlWorkSheet.Cells[17, 1] = "Avgepoches";
            xlWorkSheet.Cells[17, 2] = ret.AverageEpoch;
            xlWorkSheet.Cells[18, 1] = "Epoches";

            for (int i = 0; i < ret.Epoches.Count; i++)
            {
                xlWorkSheet.Cells[19 + i, 1] = i + 1;
                xlWorkSheet.Cells[19 + i, 2] = ret.Epoches[i];
            }

            xlWorkSheet.Columns.AutoFit();
            xlWorkSheet.UsedRange.HorizontalAlignment = Excel.XlHAlign.xlHAlignCenter;

            Marshal.ReleaseComObject(xlWorkSheet);
        }
       
        public class Metrics
        {
            public double Recall { get; set; }
            public double Precision { get; set; }
            public double F1Score { get; set; }
        }

        public class Result
        {
            public string Name { get; set; }

            public Tuple<int, string> Feature { get; set; }

            public double AverageCSR { get; set; }

            public List<int> Epoches = new List<int>();
            public int AverageEpoch { get; set; }

            public double AverageTestSpeed { get; set; }

            public List<double> TestSpeeds = new List<double>();

            public int AverageParamCount { get; set; }

            public List<int> ParamCounts = new List<int>();

            public Dictionary<int, Metrics> Metric = new Dictionary<int, Metrics>();

            public Dictionary<int, Dictionary<int, double>> Confusion = new Dictionary<int, Dictionary<int, double>>();
        }

        public static string[] GetDirectoryFiles(string directoryPath, string extension, bool containSub = true)
        {
            if (!Directory.Exists(directoryPath)) return null;
            List<string> s = new List<string>();
            DirectoryInfo dir = new DirectoryInfo(directoryPath);
            FileSystemInfo[] infos = dir.GetFileSystemInfos();
            foreach (FileSystemInfo f in infos)
            {
                try
                {
                    if (containSub)
                    {
                        if (f.Attributes.HasFlag(FileAttributes.Directory))
                        {
                            s.AddRange(GetDirectoryFiles(f.FullName, extension));
                        }
                        else
                        {
                            if (f.Extension == extension)
                                s.Add(f.FullName);
                        }
                    }
                    else
                    {
                        if (f.Extension == extension)
                            s.Add(f.FullName);
                    }
                }
                catch
                {
                    continue;
                }
            }
            return s.ToArray();
        }

        static List<Result> QueryResults(string resultDir, int expsnum,  string methodname)
        {
            var files = GetDirectoryFiles(resultDir, ".benchmark", true).ToList();

            List<Result> rets = new List<Result>();
            foreach (var file in files)
            {
                Result ret = null;
                if (Read(file, methodname, expsnum, out ret))
                {
                    if (ReadEpoches(Path.ChangeExtension(file, ".log"), expsnum, ref ret))
                    {
                        rets.Add(ret);
                    }
                }
            }

            return rets;
        }

        public static bool Read(string path, string methodname, int expsnum, out Result ret)
        {
            DirectoryInfo dirinfo = new DirectoryInfo(Path.GetDirectoryName(path));
            string dirname = dirinfo.Name.ToLowerInvariant();
            Tuple<int, string> featurename = null;
            if (dir2feature.TryGetValue(dirname, out featurename))
            {
                ret = new Result();
                ret.Feature = featurename;
                ret.Name = featurename.Item2 + "_" + methodname;

                using (var fs = File.OpenRead(path))
                {
                    using (var reader = new BinaryReader(fs))
                    {
                        /*skip avgtestspeed for manual cal*/
                        reader.ReadInt32();
                        ret.AverageParamCount = reader.ReadInt32();

                        for (int i = 0; i < expsnum; i++)
                        {
                            ret.TestSpeeds.Add(reader.ReadDouble());
                        }

                        for (int i = 0; i < expsnum; i++)
                        {
                            ret.ParamCounts.Add((int)reader.ReadDouble());
                        }

                        ret.AverageTestSpeed = ret.TestSpeeds.Sum() / (double)ret.TestSpeeds.Count;
                        ret.AverageCSR = reader.ReadDouble();

                        for (int i = 1; i <= classnum; i++)
                        {
                            Dictionary<int, double> subconfusion = new Dictionary<int, double>();
                            for (int j = 1; j <= classnum; j++)
                            {
                                subconfusion.Add(j, reader.ReadDouble());
                            }
                            ret.Confusion.Add(i, subconfusion);
                        }

                        for (int i = 1; i <= classnum; i++)
                        {
                            ret.Metric.Add(i, new Metrics() { Recall = reader.ReadDouble() });
                        }

                        for (int i = 1; i <= classnum; i++)
                        {
                            ret.Metric[i].Precision = reader.ReadDouble();
                        }

                        for (int i = 1; i <= classnum; i++)
                        {
                            ret.Metric[i].F1Score = reader.ReadDouble();
                        }
                    }
                }
                return true;
            }
            ret = null;
            return false;
        }

        public static bool ReadEpoches(string path, int expsnum, ref Result ret)
        {
            
            using (var fs = File.OpenRead(path))
            {
                using (var reader = new BinaryReader(fs))
                {
                    for (int i = 0; i < expsnum; i++)
                    {
                        ret.Epoches.Add(reader.ReadInt32());
                    }
                    ret.AverageEpoch = reader.ReadInt32();
                }
            }
            return true;
        }

    }
}
