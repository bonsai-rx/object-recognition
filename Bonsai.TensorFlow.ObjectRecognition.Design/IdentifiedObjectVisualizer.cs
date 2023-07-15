using Bonsai;
using Bonsai.Vision.Design;
using Bonsai.TensorFlow.ObjectRecognition;
using Bonsai.TensorFlow.ObjectRecognition.Design;
using OpenTK.Graphics;
using OpenTK.Graphics.OpenGL;
using System;
using System.Windows.Forms;
using System.Collections.Generic;

[assembly: TypeVisualizer(typeof(IdentifiedObjectVisualizer), Target = typeof(IdentifiedObject))]

namespace Bonsai.TensorFlow.ObjectRecognition.Design
{
    /// <summary>
    /// Provides a type visualizer that draws a visual representation of
    /// an identified objects extracted from each image in the sequence.
    /// </summary>
    public class IdentifiedObjectVisualizer : IplImageVisualizer
    {
        IdentifiedObject identifiedObject;
        LabeledImageLayer labeledImage;
        ToolStripButton drawLabelsButton;

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
            identifiedObject = (IdentifiedObject)value;
            base.Show(identifiedObject?.Image);
        }

        /// <inheritdoc/>
        protected override void ShowMashup(IList<object> values)
        {
            base.ShowMashup(values);
            if (identifiedObject != null)
            {
                if (DrawLabels)
                {
                    labeledImage.UpdateLabels(identifiedObject.Image.Size, VisualizerCanvas.Font, (graphics, labelFont) =>
                    {
                        DrawingHelper.DrawLabels(graphics, labelFont, identifiedObject);
                    });
                }
                else labeledImage.ClearLabels();
            }
        }

        /// <inheritdoc/>
        protected override void RenderFrame()
        {
            GL.Color4(Color4.White);
            base.RenderFrame();

            if (identifiedObject != null)
            {
                DrawingHelper.SetDrawState(VisualizerCanvas);
                DrawingHelper.DrawIdentifiedObject(identifiedObject);
                labeledImage.Draw();
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
