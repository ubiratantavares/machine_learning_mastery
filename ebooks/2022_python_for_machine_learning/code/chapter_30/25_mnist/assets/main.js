function pageinit() {
	// Set up canvas object
	var canvas = document.getElementById("writing");
	canvas.width = parseInt($("#writing").css("width"));
	canvas.height = parseInt($("#writing").css("height"));
	var context = canvas.getContext("2d");  // to remember drawing
	context.strokeStyle = "#FF0000";        // draw in bright red
	context.lineWidth = canvas.width / 15;  // thickness adaptive to canvas size

	// Canvas reset by timeout
	var timeout = null; // holding the timeout event
	var reset = function() {
		// clear the canvas
		context.clearRect(0, 0, canvas.width, canvas.height);
	}

	// Set up drawing with mouse
	var mouse = {x:0, y:0}; // to remember the coordinate w.r.t. canvas
	var onPaint = function() {
		clearTimeout(timeout);
		// event handler for mouse move in canvas
		context.lineTo(mouse.x, mouse.y);
		context.stroke();
	};

	// HTML5 Canvas mouse event - in case of desktop browser
	canvas.addEventListener("mousedown", function(e) {
		clearTimeout(timeout);
		// mouse down, begin path at mouse position
		context.moveTo(mouse.x, mouse.y);
		context.beginPath();
		// all mouse move from now on should be painted
		canvas.addEventListener("mousemove", onPaint, false);
	}, false);
	canvas.addEventListener("mousemove", function(e) {
		// mouse move remember position w.r.t. canvas
		mouse.x = e.pageX - this.offsetLeft;
		mouse.y = e.pageY - this.offsetTop;
	}, false);
	canvas.addEventListener("mouseup", function(e) {
		// all mouse move from now on should NOT be painted
		canvas.removeEventListener("mousemove", onPaint, false);
		clearTimeout(timeout);
		// read drawing into image
		var img = new Image(); // on load, this will be the canvas in same WxH
		img.onload = function() {
			// Draw the 28x28 to top left corner of canvas
			context.drawImage(img, 0, 0, 28, 28);
			// Extract data: Each pixel becomes a RGBA value, hence 4 bytes each
			var data = context.getImageData(0, 0, 28, 28).data;
			var input = [];
			for (var i=0; i<data.length; i += 4) {
				// scan each pixel, extract first byte (R component)
				input.push(data[i]);
			};
			var matrix = [];
			for (var i=0; i<input.length; i+=28) {
				matrix.push(input.slice(i, i+28).toString());
			};
			$("#lastinput").html("[[" + matrix.join("],\n[") + "]]");
			// call predict function with the matrix
			predict(input);
		};
		img.src = canvas.toDataURL("image/png");
		timeout = setTimeout(reset, 5000); // clear canvas after 5 sec
	}, false);

	function predict(input) {
		$.ajax({
			type: "POST",
			url: "/recognize",
			data: {"matrix": JSON.stringify(input)},
			success: function(result) {
				$("#predictresult").html(result);
			}
		});
	};
};
