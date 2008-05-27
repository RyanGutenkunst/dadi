python module tridiag
interface
  subroutine tridiag(a, b, c, r, u, n)
    intent(c) tridiag
    intent(c)        
    double precision intent(in), dimension(n) :: a
    double precision intent(in), dimension(n) :: b
    double precision intent(in), dimension(n) :: c
    double precision intent(in), dimension(n) :: r
    double precision intent(out), dimension(n) :: u
    integer intent(hide), depend(r) :: n=len(r)
  end subroutine tridiag
  subroutine tridiag_fl(a, b, c, r, u, n)
    intent(c) tridiag_fl
    intent(c)        
    real intent(in), dimension(n) :: a
    real intent(in), dimension(n) :: b
    real intent(in), dimension(n) :: c
    real intent(in), dimension(n) :: r
    real intent(out), dimension(n) :: u
    integer intent(hide), depend(r) :: n=len(r)
  end subroutine tridiag_fl
end interface
end python module tridiag
