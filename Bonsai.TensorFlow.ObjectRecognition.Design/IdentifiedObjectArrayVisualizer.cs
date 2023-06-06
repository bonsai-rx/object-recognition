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
    public class IdentifiedObjectArrayVisualizer : IplImageVisualizer
    {
        List<IdentifiedObject> idedObjectArray;
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
            idedObjectArray = value as List<IdentifiedObject>;
            if (idedObjectArray.Count > 0) {
                base.Show(idedObjectArray[0]?.Image); 
            }
        }

        protected override void ShowMashup(IList<object> values)
        {
            base.ShowMashup(values);
            var image = VisualizerImage;

            if ((idedObjectArray != null))
            {
                if (idedObjectArray.Count > 0)
                {
                    if (DrawLabels)
                    {
                        labeledImage.UpdateLabels(idedObjectArray[0].Image.Size, VisualizerCanvas.Font, (graphics, labelFont) =>
                        {
                            foreach(var idedObject in idedObjectArray)
                            {
                                DrawingHelper.DrawLabels(graphics, labelFont, idedObject);
                            }
                        });
                    }
                    else labeledImage.ClearLabels();
                }
            }
        }

        protected override void RenderFrame()
        {
            GL.Color4(Color4.White);
            base.RenderFrame();

            if ((idedObjectArray != null))
            {
                if (idedObjectArray.Count > 0)
                {
                    DrawingHelper.SetDrawState(VisualizerCanvas);
                    int i = 0;
                    foreach (var idedObject in idedObjectArray)
                    {
                        DrawingHelper.DrawIdentifiedObject(idedObject, i);
                        i++;
                    }
                    labeledImage.Draw();
                }
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
