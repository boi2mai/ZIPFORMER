`timescale 1ns/1ps

module tb_AXI_master;

parameter ADDR_WIDTH = 32;
parameter DATA_WIDTH = 128;
parameter MEM_DEPTH  = 256;

logic clk;
logic reset_n;

////////////////////////////////////////
// AXI signals
////////////////////////////////////////

logic awready;
logic awvalid;
logic [ADDR_WIDTH-1:0] awaddr;

logic wready;
logic wvalid;
logic [DATA_WIDTH-1:0] wdata;
logic wlast;

logic bvalid;
logic [1:0] bresp;
logic bready;

logic arready;
logic arvalid;
logic [ADDR_WIDTH-1:0] araddr;

logic rvalid;
logic [DATA_WIDTH-1:0] rdata;
logic [1:0] rresp;
logic rlast;
logic rready;

////////////////////////////////////////
// control
////////////////////////////////////////

logic start_write;
logic start_read;

logic [ADDR_WIDTH-1:0] addr_write;
logic [ADDR_WIDTH-1:0] addr_read;
logic [DATA_WIDTH-1:0] data_write;

logic [DATA_WIDTH-1:0] data_out;
logic valid;

////////////////////////////////////////
// memory
////////////////////////////////////////

logic [DATA_WIDTH-1:0] mem [0:MEM_DEPTH-1];

////////////////////////////////////////
// DUT
////////////////////////////////////////

AXI_master dut(

.clk(clk),
.reset_n(reset_n),

.awready(awready),
.awlen(8'd0),
.awvalid(awvalid),
.awaddr(awaddr),
.awprot(),
.awsize(),
.awburst(),

.wready(wready),
.wvalid(wvalid),
.wdata(wdata),
.wstrb(),
.wlast(wlast),

.bvalid(bvalid),
.bresp(bresp),
.bready(bready),

.arready(arready),
.arvalid(arvalid),
.araddr(araddr),
.arprot(),
.arlen(),
.arsize(),
.arburst(),

.rvalid(rvalid),
.rdata(rdata),
.rresp(rresp),
.rlast(rlast),
.rready(rready),

.start_write(start_write),
.start_read(start_read),
.addr_read(addr_read),
.addr_write(addr_write),
.data_write(data_write),

.data_out(data_out),
.done(),
.valid(valid),
.status()

);

////////////////////////////////////////
// CLOCK
////////////////////////////////////////

always #5 clk = ~clk;

////////////////////////////////////////
// RESET
////////////////////////////////////////

initial begin
    clk = 0;
    reset_n = 0;

    start_write = 0;
    start_read  = 0;

    awready = 0;
    wready  = 0;
    bvalid  = 0;

    arready = 0;
    rvalid  = 0;
    rlast   = 0;

    repeat(5) @(posedge clk);
    reset_n = 1;
end

////////////////////////////////////////
// RANDOM READY GENERATOR
////////////////////////////////////////

always @(posedge clk) begin
    awready <= $urandom_range(0,1);
    wready  <= $urandom_range(0,1);
    arready <= $urandom_range(0,1);
end

////////////////////////////////////////
// WRITE SLAVE
////////////////////////////////////////

always @(posedge clk) begin

    if(awvalid && awready) begin
        $display("WRITE ADDR %h", awaddr);
    end

    if(wvalid && wready) begin

        mem[awaddr>>4] <= wdata;

        if(wlast) begin
            bvalid <= 1;
            bresp  <= 0;
        end
    end

    if(bvalid && bready)
        bvalid <= 0;

end

////////////////////////////////////////
// READ SLAVE
////////////////////////////////////////

always @(posedge clk) begin

    if(arvalid && arready) begin

        rvalid <= 1;
        rdata  <= mem[araddr>>4];
        rlast  <= 1;
        rresp  <= 0;

        $display("READ ADDR %h", araddr);

    end

    if(rvalid && rready) begin
        rvalid <= 0;
        rlast  <= 0;
    end

end

////////////////////////////////////////
// WRITE TASK
////////////////////////////////////////

task axi_write(input [31:0] addr, input [127:0] data);

begin

    @(posedge clk);

    addr_write = addr;
    data_write = data;

    start_write = 1;
    @(posedge clk);
    start_write = 0;

    wait(bvalid);

    @(posedge clk);

end

endtask

////////////////////////////////////////
// READ TASK
////////////////////////////////////////

task axi_read(input [31:0] addr);

begin

    @(posedge clk);

    addr_read = addr;

    start_read = 1;
    @(posedge clk);
    start_read = 0;

    wait(valid);

end

endtask
////////////////////////////////////////
// TEST REPORT
////////////////////////////////////////

int pass_count = 0;
int fail_count = 0;

task check(input bit cond, input string name);
begin
    if(cond) begin
        pass_count++;
        $display("[PASS] %s", name);
    end
    else begin
        fail_count++;
        $display("[FAIL] %s", name);
    end
end
endtask
/////////////////////////////////////////
// TEST SEQUENCE
////////////////////////////////////////

initial begin

    wait(reset_n);

    repeat(5) @(posedge clk);

    $display("\n==============================");
    $display("        AXI TEST PLAN");
    $display("==============================");

    //////////////////////////////////////////////////
    // WRITE TEST
    //////////////////////////////////////////////////

    $display("\n[TC1] WRITE ADDR 0x10");

    axi_write(32'h10,128'hAAAA_BBBB_CCCC_DDDD_1111_2222_3333_4444);

    check(mem[32'h10>>4] == 128'hAAAA_BBBB_CCCC_DDDD_1111_2222_3333_4444,
          "WRITE VERIFY 0x10");



    $display("\n[TC2] WRITE ADDR 0x20");

    axi_write(32'h20,128'hDEAD_BEEF_1234_5678_ABCD_EF00_5555_AAAA);

    check(mem[32'h20>>4] == 128'hDEAD_BEEF_1234_5678_ABCD_EF00_5555_AAAA,
          "WRITE VERIFY 0x20");


    repeat(10) @(posedge clk);


    //////////////////////////////////////////////////
    // READ TEST
    //////////////////////////////////////////////////

    $display("\n[TC3] READ ADDR 0x10");

    axi_read(32'h10);

    check(data_out == 128'hAAAA_BBBB_CCCC_DDDD_1111_2222_3333_4444,
          "READ VERIFY 0x10");


    $display("\n[TC4] READ ADDR 0x20");

    axi_read(32'h20);

    check(data_out == 128'hDEAD_BEEF_1234_5678_ABCD_EF00_5555_AAAA,
          "READ VERIFY 0x20");


    repeat(20) @(posedge clk);


    //////////////////////////////////////////////////
    // SUMMARY
    //////////////////////////////////////////////////

    $display("\n==============================");
    $display("        TEST SUMMARY");
    $display("==============================");

    $display("PASS = %0d", pass_count);
    $display("FAIL = %0d", fail_count);

    if(fail_count == 0)
        $display("ALL TEST PASSED");
    else
        $display("SOME TEST FAILED");

    $display("==============================");

    $stop;

end
endmodule