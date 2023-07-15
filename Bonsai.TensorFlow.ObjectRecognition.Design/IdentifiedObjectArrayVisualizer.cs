using Bonsai;
using Bonsai.Vision.Design;
using Bonsai.TensorFlow.ObjectRecognition;
using Bonsai.TensorFlow.ObjectRecognition.Design;
using OpenTK.Graphics;
using OpenTK.Graphics.OpenGL;
using System;
using System.Windows.Forms;
using System.Collections.Generic;

[assembly: TypeVisualizer(typeof(IdentifiedObjectArrayVisualizer), Target = typeof(List<IdentifiedObject>))]

namespace Bonsai.TensorFlow.ObjectRecognition.Design
{
    /// <summary>
    /// Provides a type visualizer that draws a visual representation of the
    /// collection of identified objects extracted from each image in the sequence.
    /// </summary>
    public class IdentifiedObjectArrayVisualizer : IplImageVisualizer
    {
        List<IdentifiedObject> identifiedObject;
        LabeledImageLayer labeledImage;
        ToolStripButton drawLabelsButton;

        /// <summary>
        /// Gets or sets a value indicating whether to show the names
        /// of the identified objects.
        /// </summary>
        public bool DrawLabels { get; set; } = true;

        /// <inheritdoc/>
        public override void Load(IServiceProvider provider)
        {
            base.Load(provider);
            drawLabelsButton = new ToolStripButton("Draw Labels");
            drawLabelsButton.CheckState = CheckState.Checked;
            drawLabelsButton.Checked = DrawLabels;
            drawLabelsButton.CheckOnClick = true;
            drawLabelsButton.CheckedChanged += (sender, e) => DrawLabels = drawLabelsButton.Checked;
            StatusStrip.Items.Add(drawLabelsButton);

            VisualizerCanvas.Load += (sender, e) =>
            {
                labeledImage = new LabeledImageLayer();
                GL.Enable(EnableCap.PointSmooth);
            };
        }

        /// <inheritdoc/>
        public override void Show(object value)
        {
            identifiedObject = value as List<IdentifiedObject>;
            if (identifiedObject.Count > 0) {
                base.Show(identifiedObject[0]?.Image); 
            }
        }

        /// <inheritdoc/>
        protected override void ShowMashup(IList<object> values)
        {
            base.ShowMashup(values);
            var image = VisualizerImage;

            if ((identifiedObject != null))
            {
                if (identifiedObject.Count > 0)
                {
                    if (DrawLabels)
                    {
                        labeledImage.UpdateLabels(identifiedObject[0].Image.Size, VisualizerCanvas.Font, (graphics, labelFont) =>
                        {
                            foreach(var idedObject in identifiedObject)
                            {
                                DrawingHelper.DrawLabels(graphics, labelFont, idedObject);
                            }
                        });
                    }
                    else labeledImage.ClearLabels();
                }
            }
        }

        /// <inheritdoc/>
        protected override void RenderFrame()
        {
            GL.Color4(Color4.White);
            base.RenderFrame();

            if ((identifiedObject != null))
            {
                if (identifiedObject.Count > 0)
                {
                    DrawingHelper.SetDrawState(VisualizerCanvas);
                    int i = 0;
                    foreach (var idedObject in identifiedObject)
                    {
                        DrawingHelper.DrawIdentifiedObject(idedObject, i);
                        i++;
                    }
                    labeledImage.Draw();
                }
            }
        }

        /// <inheritdoc/>
        public override void Unload()
        {
            base.Unload();
            labeledImage?.Dispose();
            labeledImage = null;
        }
    }
}
