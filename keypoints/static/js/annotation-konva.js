function addHumanGroup(stage, polygonLayer, joints) {
    const JOINT_NAMES = [
        "Pelvis","L_Hip","R_Hip","Spine1","L_Knee",
        "R_Knee","Spine2","L_Ankle","R_Ankle","Spine3",
        "L_Foot","R_Foot","Neck","L_Collar","R_Collar",
        "Head","L_Shoulder","R_Shoulder","L_Elbow","R_Elbow",
        "L_Wrist","R_Wrist","L_Hand","R_Hand"
    ];
    const JOINT_CONNECT = [
        [0,1], [1,4], [4,7], [7,10], [0,2],
        [2,5], [5,8], [8,11], [0,3], [3,6],
        [6,9], [9,14], [14,17], [17,19], [19, 21],
        [21,23], [9,13], [13,16], [16,18], [18,20],
        [20,22], [9,12], [15,12]
    ]
    function addCircle(stage, x, y, name, color) {
        let anchor = new Konva.Circle({
            x: x,
            y: y,
            fill: color,
            radius: 5,
            draggable: true,
            name: name
        });
        anchor.on('mouseover', function () {
            stage.container().style.cursor = 'pointer'
        })
        anchor.on('mouseout', function() {
            stage.container().style.cursor = 'default';
        });
        return anchor;
    }
    function addLine(stage, start, end, name, color) {
        let anchor = new Konva.Line({
            name: name,
            stroke: 'red',
            strokeWidth: 5,
            lineCap: 'round',
            lineJoin: 'round',
        });
        anchor.points([start[0], start[1], end[0], end[1]]);
        return anchor;
    }
    // constructor
    let human = new Konva.Group();
    human.name('human');
    // add lines
    $.each(JOINT_CONNECT, function (i, item) {
        let start = joints[item[0]];
        let end = joints[item[1]];
        human.add(addLine(stage, start, end, 'line'+i, '#47C83E'));
    });
    // add circle
    $.each(joints, function (i, item) {
        let pX = item[0];
        let pY = item[1]
        human.add(addCircle(stage, pX, pY, 'circle'+i, '#47C83E'));
    });
    polygonLayer.add(human);

    // add event
    $.each(human.find('Circle'), function (i, item) {
       item.on('dragmove', function (event) {
           let target = $(this)[0];
           let connectedTargets = [];
           let circleIdx = parseInt(target.name().replace('circle', ''));
           $.each(JOINT_CONNECT, function (i, item) {
               if (item.includes(circleIdx))
                   connectedTargets.push(i)
           });
           let line, start, end;
           $.each(connectedTargets, function (i, item) {
               line = human.findOne('.line'+item);
               start = human.findOne('.circle'+JOINT_CONNECT[item][0]);
               end = human.findOne('.circle'+JOINT_CONNECT[item][1]);
               line.points([start.x(), start.y(), end.x(), end.y()]);
               polygonLayer.batchDraw();
           });
       })
    });

    return human;
}