using Bonsai.Vision.Design;
using Bonsai.TensorFlow.ObjectRecognition;
using OpenCV.Net;
using OpenTK;
using OpenTK.Graphics.OpenGL;
using Font = System.Drawing.Font;
using Graphics = System.Drawing.Graphics;
using Brushes = System.Drawing.Brushes;

namespace Bonsai.TensorFlow.ObjectRecognition.Design
{
    internal static class DrawingHelper
    {
        public static Vector2 NormalizePoint(Point2f point, Size imageSize)
        {
            return new Vector2(
                (point.X * 2f / imageSize.Width) - 1,
              -((point.Y * 2f / imageSize.Height) - 1));
        }

        public static void SetDrawState(VisualizerCanvas canvas)
        {
            const float BoundingBoxLineWidth = 3;
            GL.PointSize(5 * canvas.Height / 480f);
            GL.LineWidth(BoundingBoxLineWidth);
            GL.Disable(EnableCap.Texture2D);
        }

        public static void DrawIdentifiedObject(IdentifiedObject idedObject, int colorIndex = 0)
        {
            var imageSize = idedObject.Image.Size;
            Point2f[] roiLimits =
            {
                new Point2f(idedObject.Box.LowerLeft.X, idedObject.Box.LowerLeft.Y),
                new Point2f(idedObject.Box.UpperRight.X, idedObject.Box.LowerLeft.Y),
                new Point2f(idedObject.Box.UpperRight.X, idedObject.Box.UpperRight.Y),
                new Point2f(idedObject.Box.LowerLeft.X, idedObject.Box.UpperRight.Y)
            };
            GL.Color3(ColorPalette.GetColor(colorIndex));
            GL.Begin(PrimitiveType.LineLoop);
            for (int i = 0; i < roiLimits.Length; i++)
            {
                GL.Vertex2(NormalizePoint(roiLimits[i], imageSize));
            }
            GL.End();
        }

        public static void DrawLabels(Graphics graphics, Font font, IdentifiedObject idedObject)
        {
            if (!string.IsNullOrEmpty(idedObject.Name))
            {
                var _label = string.Format("{0} ({1:0.##}%)",
                    idedObject.Name, idedObject.Confidence * 100.0);
                graphics.DrawString(
                    _label,
                    font,
                    Brushes.White,
                    idedObject.Box.LowerLeft.X,
                    idedObject.Box.LowerLeft.Y);
            }
        }
    }
}
