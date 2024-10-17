use opencv::Result;

use crate::util::BoxDetection;

use super::yolov8::Yolov8;

pub enum ObjectDetectionModel{
    Yolov8(Yolov8)
    //... add another model
}
impl ObjectDetectionModel{
    pub fn detect(&self, input: &[f32]) -> Result<Vec<BoxDetection>, Box<dyn std::error::Error>>{
        match self{
            ObjectDetectionModel::Yolov8(m) => m.detect(input),
        }
    }
}