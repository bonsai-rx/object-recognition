using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Reactive.Linq;
using System.Reflection;
using Bonsai.Expressions;
using OpenCV.Net;
using TensorFlow;

namespace Bonsai.TensorFlow.ObjectRecognition
{
    /// <summary>
    /// Represents an operator that performs object recognition on an RGB image by
    /// running inference on a "ssd_inception_v2_coco_2017_11_17" network.
    /// </summary>
    [Description("Performs object recognition with a ssd_inception_v2_coco_2017_11_17 network.")]
    public class PredictObject : Transform<IplImage, List<IdentifiedObject>>
    {
        /// <summary>
        /// Maximum number of returned identified objects, sorted by confidence. If empty,
        /// the operator will return all identified objects in the image.
        /// </summary>
        public int? TopHits { get; set; }

        /// <summary>
        /// Gets or sets a value specifying the confidence threshold used to discard predicted
        /// objects identities.
        /// </summary>
        [Range(0, 1)]
        [Editor(DesignTypes.SliderEditor, DesignTypes.UITypeEditor)]
        [Description("Specifies the confidence threshold used to discard predicted body part positions. If no value is specified, all estimated positions are returned.")]
        public float MinimumConfidence { get; set; } = 0;

        /// <summary>
        /// Performs image object recognition for each array of images in an observable sequence
        /// using a "ssd_inception_v2_coco_2017_11_17" network.
        /// </summary>
        /// <param name="source">The sequence of image batches from which to extract the identities.</param>
        /// <returns>
        /// A sequence of <see cref="IdentifiedObject"/> objects representing the results
        /// object identification for each image batch in the <paramref name="source"/>
        /// sequence.
        /// </returns>
        private IObservable<List<IdentifiedObject>> Process(IObservable<IplImage[]> source)
        {
            return Observable.Defer(() =>
            {
                IplImage resizeTemp = null;
                TFTensor tensor = null;
                TFSession.Runner runner = null;
                var classLabels = ExtensionMethods.GetClassLabels();

                var basePath = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
                const string ModelName = "ssd_inception_v2_coco_2017_11_17.pb";
                var defaultPath = Path.Combine(basePath, ModelName);

                if (!File.Exists(defaultPath)) defaultPath = Path.Combine(basePath, "..\\..\\content\\", ModelName);
                var graph = TensorHelper.ImportModel(defaultPath, out TFSession session);
                var labels = ExtensionMethods.GetClassLabels();

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
                        runner.Fetch(graph["Identity_1"][0]); // Index of class labels
                        runner.Fetch(graph["Identity_2"][0]); // Ordered confidence (descending)
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
                    var topHitsToTake = ((TopHits.HasValue) ? BoundingBoxArr.GetLength(1) : TopHits.Value);
                    if (topHitsToTake > 0)
                    {
                        for (int i = 0; i < topHitsToTake; i++)
                        {
                            var identifiedObject = new IdentifiedObject(input[0]);
                            identifiedObject.Name = labels[(int)LabelIdxArr[batch_idx, i] - 1];

                            var box = new BoundingBox();
                            var width = identifiedObject.Image.Width;
                            var height = identifiedObject.Image.Height;
                            box.LowerLeft = new Point2f(
                                BoundingBoxArr[batch_idx, i, 1] * width,
                                BoundingBoxArr[batch_idx, i, 0] * height);
                            box.UpperRight = new Point2f(
                                BoundingBoxArr[batch_idx, i, 3] * width,
                                BoundingBoxArr[batch_idx, i, 2] * height);
                            identifiedObject.Box = box;

                            identifiedObject.Confidence = ConfidenceArr[batch_idx, i];
                            topObjects.Add(identifiedObject);
                        }
                    }

                    return (topObjects
                    .Where(x => x.Confidence > MinimumConfidence)
                    .OrderBy(v => v.Confidence).ToList());
                });
            });
        }

        /// <summary>
        /// Performs image object recognition for each image in an observable sequence using a
        /// "ssd_inception_v2_coco_2017_11_17" network.
        /// </summary>
        /// <param name="source">The sequence of images from which to extract the identities.</param>
        /// <returns>
        /// A sequence of <see cref="IdentifiedObject"/> objects representing the results
        /// object identification for each image batch in the <paramref name="source"/>
        /// sequence.
        /// </returns>
        public override IObservable<List<IdentifiedObject>> Process(IObservable<IplImage> source)
        {
            return Process(source.Select(frame => new IplImage[] { frame }));
        }
    }

    /// <summary>
    /// Represents a bounding box around the identified object
    /// </summary>
    public class BoundingBox
    {
        /// <summary>
        /// Gets or sets the lower left corner of the bounding box.
        /// </summary>
        public Point2f LowerLeft;

        /// <summary>
        /// Gets or sets the upper right corner of the bounding box.
        /// </summary>
        public Point2f UpperRight;
    }

    /// <summary>
    /// Represents an identified object from an image.
    /// </summary>
    public class IdentifiedObject
    {
        /// <summary>
        /// Gets or sets the label name of the identified object.
        /// </summary>
        public string Name;

        /// <summary>
        /// Gets or sets the bounding box around the identified object.
        /// </summary>
        public BoundingBox Box;

        /// <summary>
        /// Gets or sets the confidence score for the predicted location.
        /// </summary>
        public float Confidence;

        /// <summary>
        /// Initializes a new instance of the <see cref="IdentifiedObject"/> 
        /// class extracted from the specified image.
        /// </summary>
        /// <param name="image">The image from which the object 
        /// was identified from.</param>
        public IdentifiedObject(IplImage image)
        {
            Image = image;
        }

        /// <summary>
        /// Gets the image from which the  was identified from.
        /// </summary>
        public IplImage Image { get; }
    }
}
