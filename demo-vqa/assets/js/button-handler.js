$(document).ready(function(){
    $("#btn1").click(function(){
      $('#mySelect').empty();
      var image = $('#file').val();
      var coco_id = image.substring(image.length - 15, image.length - 4);
      var i = 0;
      while (coco_id[i] == '0') i++;
      var image_id = coco_id.substring(i, coco_id.length)
      console.log(image_id);
      $.getJSON("assets/json/iid_to_qids.json", function(json) {
        var list_questions_id = json[image_id];
        list_questions = []
        $.getJSON("assets/json/qid_to_question.json", function(json1) {
              for (id in list_questions_id) list_questions.push(json1[list_questions_id[id]]);
              for (key in list_questions) {
                //console.log(key);
                $('#mySelect').append($("<option></option>")
                        .attr("value", key)
                        .text(list_questions[key])); 
              }
        });
      });

    });

    $("#btn2").click(function(){
      var option_val = $('#mySelect').val();
      var option_text = $('#mySelect').find(":selected").text();

      var image = $('#file').val();
      var coco_id = image.substring(image.length - 15, image.length - 4);
      var i = 0;
      while (coco_id[i] == '0') i++;
      var image_id = coco_id.substring(i, coco_id.length)

      $.getJSON("assets/json/iid_to_qids.json", function(json2) {
        var list_questions_id = json2[image_id]; 
        console.log(list_questions_id);
        question_id = list_questions_id[option_val];
        console.log(question_id);
        $.getJSON("assets/json/qid_to_aid.json", function(json3) {
            label = json3[question_id];
            console.log(label);
            $.getJSON("assets/json/aid_to_answer.json", function(json4) {
                answer = json4[parseInt(label)];
                $("#test1").text("The answer for the model is: " + answer);
            });
        });

      });

    });

  });