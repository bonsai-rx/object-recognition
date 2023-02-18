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
    public class IdentifiedObjectVisualizer : IplImageVisualizer
    {
        IdentifiedObject idedObject;
        LabeledImageLayer labeledImage;
        ToolStripButton drawLabelsButton;

        public bool DrawLabels { get; set; } = true;

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

        public override void Show(object value)
        {
            idedObject = (IdentifiedObject)value;
            base.Show(idedObject?.Image);
        }

        protected override void ShowMashup(IList<object> values)
        {
            base.ShowMashup(values);
            if (idedObject != null)
            {
                if (DrawLabels)
                {
                    labeledImage.UpdateLabels(idedObject.Image.Size, VisualizerCanvas.Font, (graphics, labelFont) =>
                    {
                        DrawingHelper.DrawLabels(graphics, labelFont, idedObject);
                    });
                }
                else labeledImage.ClearLabels();
            }
        }

        protected override void RenderFrame()
        {
            GL.Color4(Color4.White);
            base.RenderFrame();

            if (idedObject != null)
            {
                DrawingHelper.SetDrawState(VisualizerCanvas);
                DrawingHelper.DrawIdentifiedObject(idedObject);
                labeledImage.Draw();
            }
        }

        public override void Unload()
        {
            base.Unload();
            labeledImage?.Dispose();
            labeledImage = null;
        }
    }
}
