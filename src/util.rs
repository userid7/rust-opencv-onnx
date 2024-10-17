use opencv::core::{Mat, Rect, Scalar};

#[derive(Debug)]
pub struct BoxDetection {
    pub rect: Rect,
    pub label: String, // class index
    pub prob: f32  // confidence score
}

pub fn draw_bounding_boxes(mut frame: &mut Mat, boxes: Vec<BoxDetection>) {
    for i in boxes {
        let rect = i.rect;
        let label = i.label;

        println!("label: {:?}, rect: {:?}", label, rect);

        // change according to your needs
        if label.to_string() == "person" {
            let box_color = Scalar::new(0.0, 255.0, 0.0, 0.0); // green color
            opencv::imgproc::rectangle(&mut frame, rect, box_color, 2, opencv::imgproc::LINE_8, 0).unwrap();
        } else if label.to_string() == "apple" {
            let box_color = Scalar::new(0.0, 165.0, 255.0, 0.0); // orange color
            opencv::imgproc::rectangle(&mut frame, rect, box_color, 2, opencv::imgproc::LINE_8, 0).unwrap();
        } else {
            let box_color = Scalar::new(0.0, 165.0, 80.0, 0.0); 
            opencv::imgproc::rectangle(&mut frame, rect, box_color, 2, opencv::imgproc::LINE_8, 0).unwrap();
        }
    };
}