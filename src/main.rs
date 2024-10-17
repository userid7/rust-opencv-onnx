use model::{object_detection_model::ObjectDetectionModel, yolov8::Yolov8};
use opencv::{core, dnn, highgui,prelude::*, videoio, Result};
use util::draw_bounding_boxes;

mod util;
pub mod model;

fn main() -> Result<(), Box<dyn std::error::Error>> {
	let model = ObjectDetectionModel::Yolov8(Yolov8::new()?);

	let window = "video capture";
	highgui::named_window(window, highgui::WINDOW_AUTOSIZE)?;
	let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)?; // 0 is the default camera
	let opened = videoio::VideoCapture::is_opened(&cam)?;
	if !opened {
		panic!("Unable to open default camera!");
	}
	
	println!("Frame width: {}", cam.get(videoio::CAP_PROP_FRAME_WIDTH)?.round());
    println!("Frame height: {}", cam.get(videoio::CAP_PROP_FRAME_HEIGHT)?.round());
	loop {
		let mut frame = Mat::default();
		cam.read(&mut frame)?;

		let blob = dnn::blob_from_image(
			&frame,       // Batch of images
			1.0 / 255.0,   // Scale factor to normalize pixel values
			core::Size::new(640, 640), // Target size for network input (example: 224x224)
			core::Scalar::new(0.0, 0.0, 0.0, 0.0), // Mean subtraction values (optional)
			true,          // Swap Red and Blue channels
			false,          // No cropping, just resize
			core::CV_32F
		)?;
		
		let raw_input = blob.data_typed::<f32>()?;

		let results = model.detect(raw_input)?;

		draw_bounding_boxes(&mut frame, results);

		if frame.size()?.width > 0 {
			highgui::imshow(window, &frame)?;
		}
		let key = highgui::wait_key(10)?;
		if key > 0 && key != 255 {
			break;
		}
	}
	Ok(())
}