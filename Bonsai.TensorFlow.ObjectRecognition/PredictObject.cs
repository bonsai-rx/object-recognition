using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Reactive.Linq;
using Bonsai.Expressions;
using OpenCV.Net;
using TensorFlow;

namespace Bonsai.TensorFlow.ObjectRecognition
{
    [Description("Performs object recognition with a ssd_inception_v2_coco_2017_11_17 network.")]
    public class PredictObject : Transform<IplImage, IdentifiedObject[]>
    {
        /// <summary>
        /// ssd_inception_v2_coco_2017_11_17
        /// </summary>
        [FileNameFilter("Protocol Buffer Files(*.pb)|*.pb")]
        [Editor("Bonsai.Design.OpenFileNameEditor, Bonsai.Design", DesignTypes.UITypeEditor)]
        [Description("Specifies the path to the exported Protocol Buffer file containing the pretrained model.")]
        public string ModelFileName { get; set; }

        [FileNameFilter("Text Files(*.*)|*.*")]
        [Editor("Bonsai.Design.OpenFileNameEditor, Bonsai.Design", DesignTypes.UITypeEditor)]
        [Description("Specifies the path to the text file containing the labels of the network (each line a label).")]
        public string Labels { get; set; }

        public int TopHits { get; set; } = 1;

        private IObservable<IdentifiedObject[]> Process(IObservable<IplImage[]> source)
        {
            return Observable.Defer(() =>
            {
                IplImage resizeTemp = null;
                TFTensor tensor = null;
                TFSession.Runner runner = null;
                var graph = TensorHelper.ImportModel(ModelFileName, out TFSession session);

                var labels = ExtensionMethods.CsvToArray(Labels);

                return source.Select(input =>
                {
                    int colorChannels = input[0].Channels;
                    var tensorSize = input[0].Size;
                    var batchSize = input.Length;
                    if (batchSize > 1) { throw new NotImplementedException("Batch inference  not implemented."); }

                    if (tensor == null || tensor.Shape[0] != batchSize || tensor.Shape[1] != tensorSize.Height || tensor.Shape[2] != tensorSize.Width)
                    {
                        tensor?.Dispose();
                        runner = session.GetRunner();
                        tensor = TensorHelper.CreatePlaceholder(graph, runner, tensorSize, batchSize, colorChannels, TFDataType.UInt8);

                        runner.Fetch(graph["Identity"][0]); // Bounding box
                        runner.Fetch(graph["Identity_1"][0]); // index of labels
                        runner.Fetch(graph["Identity_2"][0]); // ordered confidence (top = max)
                    }

                    var frames = Array.ConvertAll(input, frame =>
                    {
                        frame = TensorHelper.EnsureFrameSize(frame, tensorSize, ref resizeTemp);
                        return frame;
                    });
                    TensorHelper.UpdateTensor(tensor, colorChannels, Depth.U8, frames);
                    var output = runner.Run();


                    float[,,] BoundingBoxArr = new float[output[0].Shape[0], output[0].Shape[1], output[0].Shape[2]];
                    output[0].GetValue(BoundingBoxArr);
                    float[,] LabelIdxArr = new float[output[1].Shape[0], output[1].Shape[1]];
                    output[1].GetValue(LabelIdxArr);
                    float[,] ConfidenceArr = new float[output[2].Shape[0], output[2].Shape[1]];
                    output[2].GetValue(ConfidenceArr);

                    var topObjects = new List<IdentifiedObject>(); 

                    int batch_idx = 0;
                    for (int i = 0; i < TopHits; i++)
                    {
                        var idedObject = new IdentifiedObject(input[0]);
                        idedObject.Name = labels[(int)LabelIdxArr[batch_idx, i]-1];
                        
                        var box = new BoundingBox();
                        var width = idedObject.Image.Width;
                        var height = idedObject.Image.Height;
                        box.LowerLeft = new Point2f(
                            BoundingBoxArr[batch_idx, i, 1] * width,
                            BoundingBoxArr[batch_idx, i, 0] * height);
                        box.UpperRight = new Point2f(
                            BoundingBoxArr[batch_idx, i, 3] * width, 
                            BoundingBoxArr[batch_idx, i, 2] * height);
                        idedObject.Box = box;

                        idedObject.Confidence = ConfidenceArr[batch_idx, i];
                        topObjects.Add(idedObject);
                    }

                    return topObjects.ToArray();
                });
            });
        }

        public override IObservable<IdentifiedObject[]> Process(IObservable<IplImage> source)
        {
            return Process(source.Select(frame => new IplImage[] { frame }));
        }
    }

    public class BoundingBox
    {
        public Point2f LowerLeft;
        public Point2f UpperRight;
    }

    public class IdentifiedObject
    {
        public string Name;
        public BoundingBox Box;
        public float Confidence;

        public IdentifiedObject(IplImage image)
        {
            Image = image;
        }

        public IplImage Image { get; }
    }
}
