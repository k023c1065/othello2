const canvas1 = document.getElementById("canvas_board");
const canvas2 = document.getElementById("canvas_stone");
let {
    width,
    height
} = canvas1; // canvasのwidthとheightを取得
let a = width / 8; // 1マスの一辺の長さ
let turn = 1;
let bg_color = "seagreen"; // ボードの色
let board_data = Array(8).fill().map(() => Array(8).fill(0)); // ボードのデータ
function init_othello(arg_turn=1) {
    turn = arg_turn;
    
    drawBoard(); // ボードの表示
    board_data = Array(8).fill().map(() => Array(8).fill(0));
    board_data[3][3] = 1; // 黒石の初期配置
    board_data[4][4] = 1; // 黒石の初期配置
    board_data[3][4] = -1; // 白石の初期配置
    board_data[4][3] = -1; // 白石の初期
    update_board(); // ボードの更新
    if(turn == -1) {
        get_AI_move();
    }
    canvas2.addEventListener("click", click_canvas2); // クリックイベントの追加
}

function reset_board() {
    canvas2.width = canvas2.width;
}

function update_board() {
    //let ctx = canvas2.getContext("2d");
    reset_board();
    delete ctx;
    for (let i = 0; i < 8; i++) {
        for (let j = 0; j < 8; j++) {
            if (board_data[i][j] == 1) {
                drawStone(i, j, "black");
            } else if (board_data[i][j] == -1) {
                drawStone(i, j, "white");
            }
        }
    }
}

function click_canvas2(event) {
    let x = Math.floor(event.offsetX / a); // x座標
    let y = Math.floor(event.offsetY / a); // y座標
    let flg = false;
    console.log("flg:", flg);
    isValidMove([x,y],turn).then((data) => {
        flg = data["valid"];
        if (flg) {
            board_data = data["board"];
            update_board(); // ボードの更新
            document.getElementById('turn').textContent = "white";
            // /get_moveにPOSTリクエストを送信
            
        }
    });


}
function get_AI_move() {
    fetch("/get_move", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            "board": board_data,
            "turn": -turn
        }),
    })
    .then((response) => response.json())
    .then((data) => {
        board_data = data["board"];
        update_board(); // ボードの更新
        document.getElementById('turn').textContent = "black";
    });

}
async function isValidMove(move,turn) {
    // /is_valid にPOSTリクエストを送信
    let data = await fetch("/is_valid", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            "board":board_data,
            "move":move,
            "turn":turn
        }),
    });
    data = await data.json();
    return data;

}

function drawBoard() {
    let ctx = canvas1.getContext("2d"); // canvas1
    ctx.fillStyle = bg_color; // 塗る色をbg_colorに
    ctx.fillRect(0, 0, width, height); // ボードを塗る
    ctx.strokeStyle = "black"; // 線の色を黒に
    for (let i = 0; i <= 8; i++) {
        // i番目の縦線
        ctx.beginPath();
        ctx.moveTo(a * i, 0);
        ctx.lineTo(a * i, height);
        ctx.stroke(); // 線を引く
        // i番目の横線
        ctx.beginPath();
        ctx.moveTo(0, a * i);
        ctx.lineTo(width, a * i);
        ctx.stroke(); // 線を引く
    }
    drawPochi(); // 謎の黒い点
};

function drawPochi() {
    let ctx = canvas1.getContext("2d"); // canvas1
    for (let i = 2; i <= 6; i += 4) {
        for (let j = 2; j <= 6; j += 4) {
            let x = a * j; // x座標
            let y = a * i; // y座標
            let r = a * 0.08; // 半径
            ctx.beginPath(); // パスのリセット
            ctx.arc(x, y, r, 0, 2 * Math.PI, false); // 小さい円
            ctx.fillStyle = "black"; // 塗る色を黒に
            ctx.fill(); // 塗る
        }
    }
}

function drawStone(x, y, color) {
    let ctx = canvas2.getContext("2d"); // canvas2
    let r = a * 0.4; // 半径
    ctx.beginPath(); // パスのリセット
    ctx.arc(a * (x + 0.5), a * (y + 0.5), r, 0, 2 * Math.PI, false); // 大きい円
    ctx.fillStyle = color; // 塗る色をcolorに
    ctx.fill(); // 塗る

}

function drawValidMove(x,y){
    let ctx = canvas2.getContext("2d");
    ctx.fillStyle = "rgba(0, 255, 0, 0.5)";
    ctx.fillRect(a*x, a*y, a, a);
}