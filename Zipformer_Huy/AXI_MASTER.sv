`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 05/05/2025 08:31:39 AM
// Design Name: 
// Module Name: AXI_master
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module AXI_master #(
    parameter ADDR_WIDTH = 32,
    parameter DATA_WIDTH = 128
)(
    input  wire                  clk,
    input  wire                  reset_n,

    // Write Address Channel
    input  wire                  awready,
    input  [7:0]                 awlen,

    output reg                   awvalid,
    output reg [ADDR_WIDTH-1:0]  awaddr,
    output reg [2:0]             awprot,
    output reg [2:0]             awsize,
    output reg [1:0]             awburst,

    // Write Data Channel
    input  wire                  wready,

    output reg                   wvalid,
    output reg [DATA_WIDTH-1:0]  wdata,
    output reg [(DATA_WIDTH/8)-1:0] wstrb,
    output reg                   wlast,

    // Write Response Channel
    input  wire                  bvalid,
    input  wire [1:0]            bresp,

    output reg                   bready,

    // Read Address Channel
    input  wire                  arready,

    output reg                   arvalid,
    output [ADDR_WIDTH-1:0]      araddr,
    output reg [2:0]             arprot,
    output reg [7:0]             arlen,
    output reg [2:0]             arsize,
    output reg [1:0]             arburst,

    // Read Data Channel
    input  wire                  rvalid,
    input  wire [DATA_WIDTH-1:0] rdata,
    input  wire [1:0]            rresp,
    input  wire                  rlast,

    output reg                   rready,

    // Control
    input  wire                  start_write,
    input  wire                  start_read,
    input   [ADDR_WIDTH-1:0]     addr_read,
    input  wire [ADDR_WIDTH-1:0] addr_write,
    input  wire [127:0]          data_write,

    output reg  [DATA_WIDTH-1:0] data_out,
    output reg                   done,
    output reg                   valid,
    output reg [1:0]             status          

    // //Setup mode
    // input reg [7:0]             set_awlen,
    // input reg [2:0]             set_awsize,
    // input reg [1:0]             set_awburst,
    // input reg [7:0]             set_arlen,
    // input reg [2:0]             set_arsize,
    // input reg [1:0]             set_arburst                 
);

    typedef enum logic [1:0] {
        IDLE_WRITE,
        WRITE_ADDR,
        WRITE_DATA,
        WRITE_RESP
    } state_WRITE;

    typedef enum logic [1:0] {
        IDLE_READ,
        READ_ADDR,
        READ_DATA
    } state_READ;

    state_WRITE write_curr_state, write_next_state;
    state_READ read_curr_state, read_next_state;

    reg [5:0] count_write_trans; 
    logic boundary;
    always_ff @(posedge clk or negedge reset_n) begin
        if(~reset_n) begin
            boundary <= 0;
        end
        else begin
            if( read_next_state == READ_ADDR ) boundary <= 0; 
            else
                if( addr_read[11:0] == 12'hFE0 && rvalid ) boundary <= 1;
            
        end
    end
    assign araddr = addr_read;
    //valid update 
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            valid <= 0 ;
        end
        else begin
            valid <= rvalid && ~boundary ;
            data_out <= rdata ;
        end
    end

    // FSM state update
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            write_curr_state <= IDLE_WRITE;
            read_curr_state <= IDLE_READ;
        end
        else begin
            write_curr_state <= write_next_state;
            read_curr_state <= read_next_state;
        end
    end
    
    always_comb begin
        // WRITE FSM---------------------------------------------------------------------    
        unique case(write_curr_state)
            // ST_IDLE to ...
            IDLE_WRITE : begin
                if(start_write == 1) write_next_state = WRITE_ADDR;

                    else write_next_state = IDLE_WRITE ;

            end

            // ST_START_BIT
            WRITE_ADDR : begin
                if( awvalid && awready ) begin
                    write_next_state =  WRITE_DATA;
                end
                else write_next_state =  WRITE_ADDR;
            end

            // ST_DATA_BIT
            WRITE_DATA : begin
                if(wlast && wready) begin
                    write_next_state = WRITE_RESP ;
                end
                else begin
                    write_next_state = WRITE_DATA ;
                end
            end
            WRITE_RESP : begin
                if(1)
                    write_next_state = IDLE_WRITE ;
                else 
                    write_next_state = WRITE_RESP ;
            end
        endcase
        // READ FSM---------------------------------------------------------------------       
        // ST_END_TRAN 
        unique case(read_curr_state)
            IDLE_READ : begin

                    if(start_read) read_next_state = READ_ADDR ;
                    else read_next_state = IDLE_READ ;

            end

            READ_ADDR : begin
                if( arvalid && arready ) begin
                    read_next_state =  READ_DATA;
                end
                else read_next_state =  READ_ADDR;
            end

            READ_DATA : begin
                if(rlast && rready && rvalid ) begin
                    read_next_state = IDLE_READ ;
                end
                else begin
                    read_next_state = READ_DATA ;
                end
            end
        endcase
    end 
    // FSM logic
    //always_ff for output
    always_ff@(posedge clk or negedge reset_n) begin
        if(~reset_n) begin
            awaddr <= 0;
            awvalid <= 0;
            wdata <= 0;
            wvalid <= 0;
            bready <= 0;
            arvalid <= 0;
            rready <= 0;
            count_write_trans <= 0;
            status <= 0;
            wlast <= 0;
        end
        else begin
        // WRITE FSM---------------------------------------------------------------------        
            unique case( write_curr_state )
                IDLE_WRITE : begin
                    wlast <= 0;
                    if(write_next_state == WRITE_ADDR) begin
                        wdata <= data_write;
                        awaddr <= addr_write ;
                        awvalid <= 1 ;

                    end
                end
                WRITE_ADDR:begin
                    if(write_next_state == WRITE_DATA) begin
                        wvalid <= 1;
                         wlast <= 1;
                        awvalid <= 0 ;
                        //wdata <= data_in;
                        if( wvalid && wready ) begin
                        count_write_trans <= count_write_trans + 1;
                        end
                    end
                end
                WRITE_DATA: begin
                    if(write_next_state == WRITE_DATA) begin
                        wvalid <= 1;
                         wlast <= 1;
                         //wdata <= data_in;
//                        if( wvalid && wready ) begin
//                            count_write_trans <= count_write_trans + 1;
//                            case (count_write_trans)
//                                0 : wdata <= data_in[63:32];
//                                1 : wdata <= data_in[95:64];
//                                2 : wdata <= data_in[127:96];
//                                //3 : wdata <= data_in[127:96];
//                            endcase
//                        end
                        //else count_write_trans <= 1;
                        //if(count_write_trans == 3) wlast <= 1; //dev more
                        //wlast <= 1;
                    end
                    if(write_next_state == WRITE_RESP) begin
                        bready <= 1;
                        wvalid <= 0;
                        wlast <= 0;
                        count_write_trans <= 0;
                    end
                end
                // WRITE_RESP: begin
                //     if(write_next_state == IDLE_WRITE ) begin
                //         bready <= 0 ;
                //     end
                // end
            endcase
            // READ FSM---------------------------------------------------------------------
            unique case(read_curr_state)
            IDLE_READ : begin
                if(read_next_state == READ_ADDR) begin
                    arvalid <= 1 ;                
                end
            end
            READ_ADDR: begin
                if(read_next_state == READ_DATA) begin
                    arvalid <= 0 ;
                    rready <= 1;
                end
            end
            READ_DATA: begin
                if(read_next_state == READ_DATA) begin
                    //end
                end

                if(read_next_state == IDLE_READ) begin
                    if (rlast) rready <= 0;
                end
            end
            endcase
        end
    end
endmodule
  
