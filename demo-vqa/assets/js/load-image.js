function previewImage() {
    var file = document.getElementById("file").files
    if (file.length > 0) {
        var fileReader = new FileReader()

        fileReader.onload = function (event) {
            document.getElementById("preview").setAttribute("src", event.target.result)
        }

        fileReader.readAsDataURL(file[0])

    }
}