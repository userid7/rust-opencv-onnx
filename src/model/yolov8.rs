use opencv::{core::{Rect, Vector}, dnn, Result};
use ndarray::{Axis, s};
use ort::{Session, SessionOutputs, inputs, Tensor};

use crate::util::BoxDetection;


#[rustfmt::skip]
const YOLOV8_CLASS_LABELS: [&str; 80] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
	"fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
	"bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
	"sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
	"wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
	"carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
	"tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
	"book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
];

const MODEL_PATH: &str = "model/onnx/yolov8m.onnx";

pub struct Yolov8{
    model: Session,
    score_treshold: f32,
    nms_treshold: f32,
}
impl Yolov8{
    pub fn new() -> Result<Self, Box<dyn std::error::Error>>{
        let model = Session::builder()?.commit_from_file(MODEL_PATH)?;
        Ok(Self { model, score_treshold: 0.5, nms_treshold: 0.5 })
    }

    pub fn detect(&self, input: &[f32]) -> Result<Vec<BoxDetection>, Box<dyn std::error::Error>>{
        let input = Tensor::from_array(([1, 3, 640, 640], input))?;
    
        // Run YOLOv8 inference
        let outputs: SessionOutputs = self.model.run(inputs!["images" => input]?)?;
        let output = outputs["output0"].try_extract_tensor::<f32>()?.t().into_owned();

        let mut boxes = Vector::default();
        let mut scores = Vector::default();
        let mut indices = Vector::default();
        let mut labels = Vec::new();
        let mut indice = 0;

        let output = output.slice(s![.., .., 0]);
        for row in output.axis_iter(Axis(0)) {
            let row: Vec<_> = row.iter().copied().collect();
            let (class_id, prob) = row
                .iter()
                // skip bounding box coordinates
                .skip(4)
                .enumerate()
                .map(|(index, value)| (index, *value))
                .reduce(|accum, row| if row.1 > accum.1 { row } else { accum })
                .unwrap();

            if prob < 0.05 {
                continue;
            }

            println!("class_id: {:?}", class_id);
            let label = YOLOV8_CLASS_LABELS[class_id];
            let xc = row[0];
            let yc = row[1];
            let w = row[2];
            let h = row[3];
            println!("row: {:?}", &row[0..4]);

            let bbox = Rect {
                x: ((xc - w / 2.) ) as i32,
                y: ((yc - h / 2.) ) as i32,
                width: (w ) as i32,
                height: (h ) as i32,
            };
            println!("bbox: {:?}", bbox);
            boxes.push(bbox);
            scores.push(prob);
            labels.push(label.to_string());
            indices.push(indice);
            indice+=1;

        }

        println!("indices before: {:?}", indices.len());

        // do NMS
        dnn::nms_boxes(&boxes, &scores, self.score_treshold, self.nms_treshold, &mut indices, 1.0, 0)?;
        
        println!("indices: {:?}", indices.len());

        let mut final_boxes : Vec<BoxDetection> = Vec::default();
    
        for i in &indices {
            let idx: usize = i.try_into().unwrap();
            let bbox = BoxDetection{
                rect: boxes.get(idx).unwrap(),
                prob: scores.get(idx).unwrap(),
                label: labels.get(idx).unwrap().to_string()
            };
    
            final_boxes.push(bbox);
        };

        Ok((final_boxes))
    }
}

